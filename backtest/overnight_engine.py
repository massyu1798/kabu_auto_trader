"""オーバーナイト・ギャップ戦略バックテストエンジン v1.1

ロジック:
  - 日足データで引け買い → 翌営業日の寄り付きで決済
  - 寄り付きがギャップアップ: そのまま利確
  - 寄り付きがギャップダウン: -1.0%を超える損失は損切りキャップ
  - 日経225 ETF(1321.T)の翌日始値 vs 当日終値で「夜間追い風(+0.3%)」を近似

資金: ONG専用 (overnight_capital)
設定: config/overnight_config.yaml から読み込み（fallback: strategy_config.yaml）
"""

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import pandas_ta as ta
import yaml

from strategy.overnight_gap import OvernightGap


class Side(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    ticker: str
    side: Side
    entry_price: float
    exit_price: float
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    size: int
    pnl: float
    pnl_pct: float
    entry_reason: str = ""
    exit_reason: str = ""
    strategy_tag: str = "overnight_gap"


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    dates: list = field(default_factory=list)
    daily_halt_count: int = 0
    weekly_halt_count: int = 0
    monthly_halt_count: int = 0
    max_concurrent_positions: int = 0


# 夜間追い風の最低ギャップ率（条件3の代替近似: 翌日始値 >= 当日終値 × 1.003）
_NIGHT_GAP_PROXY_MIN = 0.003

# ギャップダウン損切りキャップ（-1.0%）
_GAP_DOWN_STOP_PCT = -0.01


class OvernightGapEngine:
    def __init__(
        self,
        config_path: str = "config/strategy_config.yaml",
        ong_config_path: str = "config/overnight_config.yaml",
    ):
        # overnight_config.yaml を優先して読み込み。存在しなければ strategy_config.yaml を使う
        import os
        ong_cfg: dict = {}
        if os.path.exists(ong_config_path):
            with open(ong_config_path, "r", encoding="utf-8") as f:
                ong_cfg = yaml.safe_load(f) or {}

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # --- ONG専用資金 ---
        ong_global = ong_cfg.get("global", {})
        g = self.config["global"]
        self.initial_capital = g["initial_capital"]
        self.commission_rate = ong_global.get("commission_rate", g.get("commission_rate", 0.0))
        self.slippage_rate = ong_global.get("slippage_rate", g.get("slippage_rate", 0.001))
        self.risk_per_trade = ong_global.get("risk_per_trade", g.get("risk_per_trade", 0.01))
        self.max_positions = ong_global.get("max_positions", g.get("max_positions", 3))

        # overnight_config.yaml に overnight_capital があれば直接使用、なければ割合計算
        if "overnight_capital" in ong_global:
            self.ong_capital = float(ong_global["overnight_capital"])
        else:
            alloc = self.config.get("capital_allocation", {})
            self.ong_capital = self.initial_capital * alloc.get("overnight_gap", 0.20)

        # --- リスク管理パラメータ ---
        ong_risk = ong_cfg.get("risk", {})
        self.max_daily_loss_pct = ong_risk.get(
            "max_daily_loss_pct", g.get("max_daily_loss", 0.025)
        )
        self.weekly_max_loss_pct = ong_risk.get(
            "weekly_max_loss_pct", g.get("weekly_max_loss", 0.05)
        )
        self.monthly_max_loss_pct = ong_risk.get(
            "monthly_max_loss_pct", g.get("monthly_max_loss", 0.10)
        )

        # --- エントリー条件パラメータ ---
        # overnight_config.yaml の entry セクションを優先
        entry_cfg = ong_cfg.get("entry", {})
        # strategy_config.yaml の overnight_gap.params も参照
        ong_params_base = (
            self.config.get("strategies", {})
            .get("overnight_gap", {})
            .get("params", {})
        )
        ong_params = {
            "ibs_oversold": entry_cfg.get(
                "ibs_oversold", ong_params_base.get("ibs_oversold", 0.3)
            ),
            "rsi2_oversold": entry_cfg.get(
                "rsi2_oversold", ong_params_base.get("rsi2_oversold", 15)
            ),
            "min_daily_drop_pct": entry_cfg.get(
                "min_daily_drop_pct", ong_params_base.get("min_daily_drop_pct", -1.0)
            ),
            "signal_score_threshold": entry_cfg.get(
                "signal_score_threshold", ong_params_base.get("signal_score_threshold", 0.9)
            ),
            "skip_fomc_boj": ong_cfg.get("filters", {}).get(
                "skip_fomc_boj", ong_params_base.get("skip_fomc_boj", True)
            ),
        }
        self.strategy = OvernightGap(ong_params)
        self.score_threshold = ong_params["signal_score_threshold"]
        self.lot_unit = 100

        # --- 夜間追い風フィルター設定 ---
        ngp = ong_cfg.get("night_gap_proxy", {})
        self._night_gap_enabled = ngp.get("enabled", True)
        self._night_gap_min = ngp.get("min_gap_pct", _NIGHT_GAP_PROXY_MIN)

        # 金曜日フィルター
        filters_cfg = ong_cfg.get("filters", {})
        self._skip_friday = filters_cfg.get("skip_friday", True)

        # 夜間追い風近似に使うETFのデータ（1321.T）
        self._nikkei_daily: pd.DataFrame | None = None

    def set_nikkei_daily(self, df: pd.DataFrame) -> None:
        """日経225 ETF(1321.T)の日足データをセット（条件3の代替近似用）"""
        self._nikkei_daily = df.copy() if df is not None else None

    def _calc_position_size(self, capital: float, entry_price: float, atr: float) -> int:
        """ATRベースのポジションサイジング（1トレード最大損失 = capital × risk_per_trade）"""
        max_loss = capital * self.risk_per_trade
        atr_mult = 2.0
        sl_distance = atr * atr_mult if (atr and atr > 0) else entry_price * 0.01
        if sl_distance <= 0:
            return 0
        size = int(max_loss / sl_distance)
        size = (size // self.lot_unit) * self.lot_unit
        # 1銘柄最大建玉キャップ（ONG配分資金の15%）
        if entry_price > 0:
            max_by_cap = int(self.ong_capital * 0.15 / entry_price // self.lot_unit) * self.lot_unit
            size = min(size, max_by_cap)
        return max(size, 0)

    def _night_gap_proxy_ok(self, date) -> bool:
        """
        夜間先物追い風フィルターの代替近似:
        日経225 ETF(1321.T)の翌日始値 >= 当日終値 × (1 + _night_gap_min)
        （実際の夜間先物+0.3%条件を翌日始値で近似）
        フィルターが無効化(enabled=False)またはデータがない場合は True（条件を満たすとみなす）
        """
        if not self._night_gap_enabled:
            return True
        if self._nikkei_daily is None:
            return True
        df = self._nikkei_daily
        # date に対応する行を探す
        date_val = date.date() if hasattr(date, "date") else date
        idx_list = [
            i for i, d in enumerate(df.index)
            if (d.date() if hasattr(d, "date") else d) == date_val
        ]
        if not idx_list:
            return True
        i = idx_list[0]
        if i + 1 >= len(df):
            return True
        today_close = float(df["close"].iloc[i])
        next_open = float(df["open"].iloc[i + 1])
        if today_close <= 0:
            return True
        return (next_open / today_close - 1.0) >= self._night_gap_min

    def run(self, daily_signals: dict[str, pd.DataFrame]) -> BacktestResult:
        """
        ONG専用バックテスト実行。

        引数:
          daily_signals: {ticker: daily_df with ong_signal-related columns}
                          daily_dfはOvernightGap.generate_signals()で計算済みのSignal Seriesと
                          元の日足データが必要。
                          実際には run() 内部でシグナル生成を行う。

        daily_signals の値は生の日足OHLCV DataFrameでも可。
        run()内でgenerate_signals()を呼び出す。
        """
        capital = self.ong_capital
        trades: list[Trade] = []
        equity_curve = []
        equity_dates = []

        # リスクトラッキング
        daily_pnl = 0.0
        weekly_pnl = 0.0
        monthly_pnl = 0.0
        current_day = None
        current_week = None
        current_month = None
        weekly_halted = False
        monthly_halted = False
        daily_halted = False
        daily_halt_count = 0
        weekly_halt_count = 0
        monthly_halt_count = 0
        max_concurrent = 0

        # シグナル生成（各銘柄の日足データに対して）
        signal_series: dict[str, pd.Series] = {}
        atr_series: dict[str, pd.Series] = {}
        for ticker, df in daily_signals.items():
            sig = self.strategy.generate_signals(df)
            signal_series[ticker] = sig
            # ATR(14) 計算
            atr = ta.atr(df["high"], df["low"], df["close"], length=14)
            atr_series[ticker] = atr if atr is not None else pd.Series(dtype=float)

        # 全取引日を収集
        all_days = sorted(set(
            (d.date() if hasattr(d, "date") else d)
            for df in daily_signals.values()
            for d in df.index
        ))

        # オープン中のポジション: {ticker: (entry_date, entry_price, size, entry_reason)}
        open_positions: dict = {}

        for day in all_days:
            # 週・月キー
            week_key = day.isocalendar()[:2] if hasattr(day, "isocalendar") else None
            month_key = (day.year, day.month) if hasattr(day, "year") else None

            if current_week != week_key:
                weekly_pnl = 0.0
                current_week = week_key
                weekly_halted = False

            if current_month != month_key:
                monthly_pnl = 0.0
                current_month = month_key
                monthly_halted = False

            if current_day != day:
                daily_pnl = 0.0
                current_day = day
                daily_halted = False

            # 週間・月間・日次停止チェック
            if not weekly_halted and abs(min(0.0, weekly_pnl)) >= self.ong_capital * self.weekly_max_loss_pct:
                weekly_halted = True
                weekly_halt_count += 1
            if not monthly_halted and abs(min(0.0, monthly_pnl)) >= self.ong_capital * self.monthly_max_loss_pct:
                monthly_halted = True
                monthly_halt_count += 1
            if not daily_halted and abs(min(0.0, daily_pnl)) >= self.ong_capital * self.max_daily_loss_pct:
                daily_halted = True
                daily_halt_count += 1

            # === 1. 前日エントリーした全ポジションを当日寄り付きで決済 ===
            positions_to_close = list(open_positions.keys())
            for ticker in positions_to_close:
                pos_info = open_positions[ticker]
                entry_date, entry_price, size, entry_reason = pos_info

                df = daily_signals.get(ticker)
                if df is None:
                    del open_positions[ticker]
                    continue

                # 当日の行を探す
                day_rows = [
                    d for d in df.index
                    if (d.date() if hasattr(d, "date") else d) == day
                ]
                if not day_rows:
                    # 当日データなし → エントリー日（前日）の終値で決済（フォールバック）
                    entry_day = entry_date.date() if hasattr(entry_date, "date") else entry_date
                    prev_rows = [
                        d for d in df.index
                        if (d.date() if hasattr(d, "date") else d) == entry_day
                    ]
                    if prev_rows:
                        exit_px = float(df.loc[prev_rows[-1], "close"])
                    else:
                        exit_px = entry_price
                    exit_date = entry_date
                    exit_reason = "データなし→前日終値"
                else:
                    exit_date = day_rows[0]
                    today_open = float(df.loc[exit_date, "open"])
                    exit_px_raw = today_open * (1 - self.slippage_rate)

                    # ギャップアップ/ダウン判定
                    gap_pct = (today_open / entry_price) - 1.0
                    if gap_pct >= 0:
                        exit_reason = f"寄り付き利確(gap+{gap_pct*100:.2f}%)"
                    elif gap_pct < _GAP_DOWN_STOP_PCT:
                        # ギャップダウン -1.0%以上 → -1.0%でキャップ
                        exit_px_raw = entry_price * (1 + _GAP_DOWN_STOP_PCT) * (1 - self.slippage_rate)
                        exit_reason = f"GapDown損切り({gap_pct*100:.2f}%→-1.0%cap)"
                    else:
                        exit_reason = f"寄り付き決済(gap{gap_pct*100:.2f}%)"

                    exit_px = exit_px_raw

                pnl = (exit_px - entry_price) * size
                pnl -= abs(exit_px * size * self.commission_rate)
                pnl_pct = pnl / (entry_price * size) * 100 if (entry_price * size) > 0 else 0.0

                trade = Trade(
                    ticker=ticker,
                    side=Side.LONG,
                    entry_price=entry_price,
                    exit_price=exit_px,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    size=size,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    entry_reason=entry_reason,
                    exit_reason=exit_reason,
                    strategy_tag="overnight_gap",
                )
                trades.append(trade)
                capital += pnl
                daily_pnl += pnl
                weekly_pnl += pnl
                monthly_pnl += pnl
                del open_positions[ticker]

            # === 2. 当日の引け時点でエントリー判定 ===
            # 日次/週間/月間損失上限チェック
            # 金曜日フィルター（overnight_config.yaml の filters.skip_friday）
            is_friday = self._skip_friday and hasattr(day, "weekday") and day.weekday() == 4
            if not (daily_halted or weekly_halted or monthly_halted) and not is_friday:
                for ticker, df in daily_signals.items():
                    if ticker in open_positions:
                        continue

                    # 同時保有上限
                    if len(open_positions) >= self.max_positions:
                        break

                    # 当日の行を探す
                    day_rows = [
                        d for d in df.index
                        if (d.date() if hasattr(d, "date") else d) == day
                    ]
                    if not day_rows:
                        continue

                    today_idx = day_rows[-1]  # 当日最終バー（終値）

                    # シグナル確認
                    sig = signal_series.get(ticker)
                    if sig is None or today_idx not in sig.index:
                        continue
                    # スカラーアクセス（duplicate index 対策: at はユニーク前提、loc はフォールバック）
                    try:
                        signal_obj = sig.at[today_idx]
                    except (KeyError, ValueError):
                        signal_obj = sig.loc[today_idx]
                    if not hasattr(signal_obj, "score"):
                        continue
                    if signal_obj.score < self.score_threshold:
                        continue

                    # 追加フィルター: 夜間追い風近似（日経225 ETF翌日始値 >= 当日終値 × min_gap_pct）
                    if not self._night_gap_proxy_ok(today_idx):
                        continue

                    row = df.loc[today_idx]
                    entry_price = float(row["close"]) * (1 + self.slippage_rate)
                    if entry_price <= 0:
                        continue
                    entry_date = today_idx

                    # ATRベースのポジションサイジング
                    atr_s = atr_series.get(ticker)
                    atr_val = 0.0
                    if atr_s is not None and today_idx in atr_s.index:
                        try:
                            v = atr_s.at[today_idx]
                        except (KeyError, ValueError):
                            v = atr_s.loc[today_idx]
                        atr_val = float(v) if (v is not None and not pd.isna(v)) else 0.0
                    if atr_val <= 0:
                        atr_val = entry_price * 0.015  # フォールバック: 1.5%

                    size = self._calc_position_size(capital, entry_price, atr_val)
                    if size < self.lot_unit:
                        continue

                    # エントリー手数料
                    capital -= abs(entry_price * size * self.commission_rate)
                    open_positions[ticker] = (
                        entry_date,
                        entry_price,
                        size,
                        f"ONG:{signal_obj.reason}",
                    )
                    max_concurrent = max(max_concurrent, len(open_positions))

            # === 3. 資産評価（未決済ポジションは含み損益なしで評価） ===
            equity_curve.append(capital)
            equity_dates.append(day)

        # 残存ポジション強制決済（バックテスト終了時）
        for ticker, pos_info in open_positions.items():
            entry_date, entry_price, size, entry_reason = pos_info
            df = daily_signals.get(ticker)
            if df is not None:
                exit_px = float(df["close"].iloc[-1]) * (1 - self.slippage_rate)
                exit_date = df.index[-1]
            else:
                exit_px = entry_price
                exit_date = entry_date
            pnl = (exit_px - entry_price) * size
            pnl -= abs(exit_px * size * self.commission_rate)
            pnl_pct = pnl / (entry_price * size) * 100 if (entry_price * size) > 0 else 0.0
            trades.append(Trade(
                ticker=ticker, side=Side.LONG,
                entry_price=entry_price, exit_price=exit_px,
                entry_date=entry_date, exit_date=exit_date,
                size=size, pnl=pnl, pnl_pct=pnl_pct,
                entry_reason=entry_reason, exit_reason="BT終了決済",
                strategy_tag="overnight_gap",
            ))
            capital += pnl

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            dates=equity_dates,
            daily_halt_count=daily_halt_count,
            weekly_halt_count=weekly_halt_count,
            monthly_halt_count=monthly_halt_count,
            max_concurrent_positions=max_concurrent,
        )
