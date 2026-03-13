"""バックテストエンジン v15: 3戦略マルチストラテジー対応
- 戦略別エグジット (MR: VWAP回帰+pct, BO: ORB復帰+分割決済, ONG: 独立ループ)
- 週間/月間損失停止ルール
- 戦略別資金配分管理
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import pandas as pd
import pandas_ta as ta
import yaml


class Side(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    ticker: str
    side: Side
    entry_price: float
    entry_date: pd.Timestamp
    size: int
    stop_loss: float
    take_profit: float
    trailing_stop: float
    reason: str = ""
    strategy_tag: str = ""                              # どの戦略由来か識別
    original_size: int = 0                              # BO分割決済用: 初期サイズ
    partial_targets: list = field(default_factory=list) # BO分割利確目標 [0.01, 0.02, 0.03]
    partial_sold: int = 0                               # 実施済み分割利確カウント
    orb_high: float = float("nan")                      # BO: ORBレンジ上限
    orb_low: float = float("nan")                       # BO: ORBレンジ下限


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
    strategy_tag: str = ""                              # 戦略タグ（MR/BO/ONG集計用）


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    dates: list = field(default_factory=list)


class BacktestEngine:
    def __init__(self, config_path: str = "config/strategy_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        g = self.config["global"]
        self.initial_capital = g["initial_capital"]
        self.max_positions = g["max_positions"]
        self.risk_per_trade = g["risk_per_trade"]
        self.max_daily_loss = g["max_daily_loss"]
        self.weekly_max_loss = g.get("weekly_max_loss", 0.05)
        self.monthly_max_loss = g.get("monthly_max_loss", 0.10)
        self.commission_rate = g["commission_rate"]
        self.slippage_rate = g["slippage_rate"]
        self.max_holding_days = g["max_holding_days"]

        e = self.config["exit"]
        self.sl_atr_mult = e["stop_loss_atr_multiplier"]
        self.tp_rr_ratio = e["take_profit_rr_ratio"]
        self.atr_period = e.get("atr_period", 14)
        self.trailing_enabled = e.get("trailing_stop", False)
        self.trailing_atr_mult = e.get("trailing_atr_multiplier", 2.0)

        cd = self.config.get("cooldown", {})
        self.cooldown_enabled = cd.get("enabled", False)
        self.cooldown_bars_loss = cd.get("bars_after_loss", 12)
        self.cooldown_bars_win = cd.get("bars_after_win", 0)

        tf = self.config.get("trend_filter", {})
        self.trend_filter_enabled = tf.get("enabled", False)
        self.trend_ema_period = tf.get("ema_period", 60)

        # 戦略別エグジット設定
        strats = self.config.get("strategies", {})
        self.mr_exit = strats.get("mean_reversion", {}).get("exit", {})
        self.bo_exit = strats.get("breakout", {}).get("exit", {})

        # 戦略別資金配分
        alloc = self.config.get("capital_allocation", {})
        self.mr_capital_limit = alloc.get("mean_reversion", 0.5) * self.initial_capital
        self.bo_capital_limit = alloc.get("breakout", 0.3) * self.initial_capital

        # リスク上限（本番との整合性）: live_risk_caps はオプショナル
        rc = self.config.get("live_risk_caps", {})
        self.max_notional_per_position = rc.get("max_notional_per_position", None)
        self.max_total_exposure = rc.get("max_total_exposure", None)
        self.safety_margin_ratio = rc.get("safety_margin_ratio", 0.98)

    def _is_market_close(self, timestamp):
        if not hasattr(timestamp, "hour"):
            return False
        return timestamp.hour >= 15 and timestamp.minute >= 20

    def _is_entry_cutoff(self, timestamp):
        if not hasattr(timestamp, "hour"):
            return False
        return timestamp.hour >= 14 and timestamp.minute >= 30

    def _is_entry_allowed(self, timestamp):
        """9:30〜14:30 をエントリー許可時間帯に変更（寄り30分の乱高下を回避）"""
        if not hasattr(timestamp, "hour"):
            return True
        t = timestamp.hour * 100 + timestamp.minute
        return 930 <= t <= 1430

    def _get_strategy_tag(self, row) -> str:
        """シグナル行の個別戦略スコアからどの戦略が支配的かを判定（v17: デフォルトMR）"""
        mr_score = abs(float(row.get("MeanReversion_score") or 0))
        bo_score = abs(float(row.get("Breakout_score") or 0))

        # BO が明確に支配的な場合のみBOタグ（MRの1.5倍以上のスコア）
        if bo_score > 0 and bo_score * 1.5 > mr_score * 2.5:
            return "breakout"

        # デフォルトはMR（主力戦略）
        return "mean_reversion"

    def _close_position(self, pos, current_price, date, reason, close_size=None):
        size = close_size if close_size is not None else pos.size
        if pos.side == Side.LONG:
            exit_price = current_price * (1 - self.slippage_rate)
            pnl = (exit_price - pos.entry_price) * size
        else:
            exit_price = current_price * (1 + self.slippage_rate)
            pnl = (pos.entry_price - exit_price) * size
        pnl -= abs(exit_price * size * self.commission_rate)
        pnl_pct = pnl / (pos.entry_price * size) * 100
        return Trade(
            ticker=pos.ticker, side=pos.side,
            entry_price=pos.entry_price, exit_price=exit_price,
            entry_date=pos.entry_date, exit_date=date,
            size=size, pnl=pnl, pnl_pct=pnl_pct,
            entry_reason=pos.reason, exit_reason=reason,
            strategy_tag=pos.strategy_tag,
        )

    def run(self, signals_dict: dict[str, pd.DataFrame]) -> BacktestResult:
        capital = self.initial_capital
        positions: list[Position] = []
        trades: list[Trade] = []
        equity_curve = []
        equity_dates = []

        for ticker, df in signals_dict.items():
            atr_col = f"ATRr_{self.atr_period}"
            if atr_col not in df.columns:
                df[atr_col] = ta.atr(
                    df["high"], df["low"], df["close"],
                    length=self.atr_period,
                )
            if self.trend_filter_enabled:
                tf_col = f"trend_ema_{self.trend_ema_period}"
                if tf_col not in df.columns:
                    df[tf_col] = ta.ema(df["close"], length=self.trend_ema_period)

        all_dates = sorted(set(
            date for df in signals_dict.values() for date in df.index
        ))

        daily_loss = 0.0
        weekly_pnl = 0.0
        monthly_pnl = 0.0
        current_day = None
        current_week = None
        current_month = None
        weekly_halted = False
        monthly_halted = False
        is_daytrade = self.max_holding_days == 0
        cooldown_until = {}
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        # MR exit config
        mr_tp_pct = self.mr_exit.get("take_profit_pct", 0.005)
        mr_sl_pct = self.mr_exit.get("stop_loss_pct", 0.003)
        mr_vwap_rev = self.mr_exit.get("vwap_reversion", True)
        mr_vwap_threshold = self.mr_exit.get("vwap_reversion_threshold", 0.3)
        mr_cutoff_str = self.mr_exit.get("time_cutoff", "14:30")
        mr_cutoff_int = int(mr_cutoff_str.replace(":", "")) if mr_cutoff_str else 1430

        # BO exit config
        bo_partial_targets = self.bo_exit.get("partial_targets", [])
        bo_sl_pct = self.bo_exit.get("stop_loss_pct", 0.007)
        bo_orb_stop = self.bo_exit.get("orb_reentry_stop", True)
        bo_cutoff_str = self.bo_exit.get("time_cutoff", "14:00")
        bo_cutoff_int = int(bo_cutoff_str.replace(":", "")) if bo_cutoff_str else 1400

        # BO entry time range from breakout strategy params
        bo_params = self.config.get("strategies", {}).get("breakout", {}).get("params", {})
        bo_entry_start_int = bo_params.get("entry_start_time", 935)
        bo_entry_end_int = bo_params.get("entry_end_time", 1300)

        for date_idx, date in enumerate(all_dates):
            day = date.date() if hasattr(date, "date") else date

            # 週/月リセット
            week_key = (day.isocalendar()[0], day.isocalendar()[1]) if hasattr(day, "isocalendar") else None
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
                if is_daytrade and positions:
                    for pos in positions:
                        if pos.ticker in signals_dict:
                            df = signals_dict[pos.ticker]
                            prev_dates = [
                                d for d in df.index
                                if (d.date() if hasattr(d, "date") else d) == current_day
                            ]
                            if prev_dates:
                                last_price = df.loc[prev_dates[-1]]["close"]
                            else:
                                last_price = pos.entry_price
                            trade = self._close_position(
                                pos, last_price,
                                prev_dates[-1] if prev_dates else date,
                                "引け強制決済",
                            )
                            trades.append(trade)
                            capital += trade.pnl
                            weekly_pnl += trade.pnl
                            monthly_pnl += trade.pnl
                    positions.clear()
                daily_loss = 0.0
                current_day = day
                cooldown_until.clear()

            # 週間/月間損失停止チェック
            if not weekly_halted and abs(min(0, weekly_pnl)) >= self.initial_capital * self.weekly_max_loss:
                weekly_halted = True
            if not monthly_halted and abs(min(0, monthly_pnl)) >= self.initial_capital * self.monthly_max_loss:
                monthly_halted = True

            is_close_time = self._is_market_close(date)
            is_cutoff = self._is_entry_cutoff(date)
            bar_time_int = date.hour * 100 + date.minute if hasattr(date, "hour") else 0

            # === 1. ポジション管理 ===
            closed_positions = []
            for pos in positions:
                if pos.ticker not in signals_dict:
                    continue
                df = signals_dict[pos.ticker]
                if date not in df.index:
                    continue

                row = df.loc[date]
                current_price = row["close"]
                high = row["high"]
                low = row["low"]
                exit_reason = None

                # --- MR戦略固有エグジット ---
                if pos.strategy_tag == "mean_reversion":
                    # 時間切り: 14:30
                    if bar_time_int >= mr_cutoff_int:
                        exit_reason = f"MR時間切り ({mr_cutoff_str})"
                    elif pos.side == Side.LONG:
                        # 損切り -0.3%
                        if current_price <= pos.entry_price * (1 - mr_sl_pct):
                            exit_reason = f"MR損切り -{mr_sl_pct*100:.1f}%"
                        # 利確 +0.5%
                        elif current_price >= pos.entry_price * (1 + mr_tp_pct):
                            exit_reason = f"MR利確 +{mr_tp_pct*100:.1f}%"
                        # VWAP回帰利確
                        elif mr_vwap_rev:
                            raw_z = row.get("vwap_z", float("nan"))
                            vwap_z_valid = not (pd.isna(raw_z) or math.isnan(float(raw_z)))
                            if vwap_z_valid and abs(float(raw_z)) <= mr_vwap_threshold:
                                exit_reason = f"MR_VWAP回帰 (z={float(raw_z):.2f})"
                    else:  # SHORT
                        if current_price >= pos.entry_price * (1 + mr_sl_pct):
                            exit_reason = f"MR損切り -{mr_sl_pct*100:.1f}%"
                        elif current_price <= pos.entry_price * (1 - mr_tp_pct):
                            exit_reason = f"MR利確 +{mr_tp_pct*100:.1f}%"
                        elif mr_vwap_rev:
                            raw_z = row.get("vwap_z", float("nan"))
                            vwap_z_valid = not (pd.isna(raw_z) or math.isnan(float(raw_z)))
                            if vwap_z_valid and abs(float(raw_z)) <= mr_vwap_threshold:
                                exit_reason = f"MR_VWAP回帰 (z={float(raw_z):.2f})"

                # --- BO戦略固有エグジット ---
                elif pos.strategy_tag == "breakout":
                    # 分割決済チェック（最後のターゲット以外は部分決済）
                    if bo_partial_targets and pos.original_size > 0:
                        target_idx = pos.partial_sold
                        if target_idx < len(bo_partial_targets):
                            tgt_pct = bo_partial_targets[target_idx]
                            tgt_price_long = pos.entry_price * (1 + tgt_pct)
                            tgt_price_short = pos.entry_price * (1 - tgt_pct)
                            hit = (
                                (pos.side == Side.LONG and current_price >= tgt_price_long)
                                or (pos.side == Side.SHORT and current_price <= tgt_price_short)
                            )
                            if hit:
                                if target_idx < len(bo_partial_targets) - 1:
                                    # 部分決済
                                    n_parts = len(bo_partial_targets)
                                    partial_size = max(100, (pos.original_size // n_parts // 100) * 100)
                                    partial_size = min(partial_size, pos.size)
                                    ptrade = self._close_position(
                                        pos, current_price, date,
                                        f"BO部分利確{target_idx+1}/{n_parts} +{tgt_pct*100:.0f}%",
                                        close_size=partial_size,
                                    )
                                    trades.append(ptrade)
                                    capital += ptrade.pnl
                                    daily_loss += min(0, ptrade.pnl)
                                    weekly_pnl += ptrade.pnl
                                    monthly_pnl += ptrade.pnl
                                    if self.cooldown_enabled and ptrade.pnl <= 0:
                                        cooldown_until[pos.ticker] = date_idx + self.cooldown_bars_loss
                                    pos.size -= partial_size
                                    pos.partial_sold += 1
                                    if pos.size <= 0:
                                        closed_positions.append(pos)
                                    continue
                                else:
                                    # 最終ターゲット到達 → 全決済
                                    exit_reason = f"BO最終利確 +{tgt_pct*100:.0f}%"

                    if exit_reason is None:
                        # BO損切り -0.7%
                        if pos.side == Side.LONG and current_price <= pos.entry_price * (1 - bo_sl_pct):
                            exit_reason = f"BO損切り -{bo_sl_pct*100:.1f}%"
                        elif pos.side == Side.SHORT and current_price >= pos.entry_price * (1 + bo_sl_pct):
                            exit_reason = f"BO損切り -{bo_sl_pct*100:.1f}%"

                    if exit_reason is None and bo_orb_stop:
                        # ORBレンジ内復帰損切り: エントリーがORBレンジ外だった場合のみ適用
                        orb_h = pos.orb_high
                        orb_l = pos.orb_low
                        if not math.isnan(orb_h) and not math.isnan(orb_l):
                            if pos.side == Side.LONG and pos.entry_price > orb_h and current_price < orb_h:
                                exit_reason = "BO_ORB復帰損切り"
                            elif pos.side == Side.SHORT and pos.entry_price < orb_l and current_price > orb_l:
                                exit_reason = "BO_ORB復帰損切り"

                # --- デフォルト（ATRベース）エグジット ---
                else:
                    # トレーリングストップ更新（加速TS）
                    if self.trailing_enabled:
                        atr_col = f"ATRr_{self.atr_period}"
                        atr_val = df[atr_col].loc[date]
                        if not pd.isna(atr_val):
                            if pos.side == Side.LONG:
                                profit_ratio = (high - pos.entry_price) / pos.entry_price
                                accel = max(0.7, 1.0 - profit_ratio * 2.0)
                                new_trail = high - atr_val * self.trailing_atr_mult * accel
                                if new_trail > pos.trailing_stop:
                                    pos.trailing_stop = new_trail
                            else:
                                profit_ratio = (pos.entry_price - low) / pos.entry_price
                                accel = max(0.7, 1.0 - profit_ratio * 2.0)
                                new_trail = low + atr_val * self.trailing_atr_mult * accel
                                if new_trail < pos.trailing_stop:
                                    pos.trailing_stop = new_trail

                    if pos.side == Side.LONG and current_price <= pos.stop_loss:
                        exit_reason = f"損切り ({pos.stop_loss:.0f})"
                    elif pos.side == Side.SHORT and current_price >= pos.stop_loss:
                        exit_reason = f"損切り ({pos.stop_loss:.0f})"
                    elif pos.side == Side.LONG and current_price >= pos.take_profit:
                        exit_reason = f"利確 ({pos.take_profit:.0f})"
                    elif pos.side == Side.SHORT and current_price <= pos.take_profit:
                        exit_reason = f"利確 ({pos.take_profit:.0f})"
                    elif self.trailing_enabled:
                        if pos.side == Side.LONG and current_price <= pos.trailing_stop:
                            exit_reason = f"TS ({pos.trailing_stop:.0f})"
                        elif pos.side == Side.SHORT and current_price >= pos.trailing_stop:
                            exit_reason = f"TS ({pos.trailing_stop:.0f})"

                # 引け強制決済（全戦略共通）
                if is_daytrade and is_close_time and exit_reason is None:
                    exit_reason = "引け強制決済"

                # 反対シグナル（MR/BO以外のデフォルト）
                if exit_reason is None and pos.strategy_tag not in ("mean_reversion", "breakout"):
                    if pos.side == Side.LONG and row.get("final_signal") == "SELL":
                        exit_reason = "反対シグナル"
                    elif pos.side == Side.SHORT and row.get("final_signal") == "BUY":
                        exit_reason = "反対シグナル"

                if exit_reason:
                    trade = self._close_position(pos, current_price, date, exit_reason)
                    trades.append(trade)
                    capital += trade.pnl
                    daily_loss += min(0, trade.pnl)
                    weekly_pnl += trade.pnl
                    monthly_pnl += trade.pnl
                    closed_positions.append(pos)

                    if self.cooldown_enabled:
                        if trade.pnl <= 0:
                            cooldown_until[pos.ticker] = date_idx + self.cooldown_bars_loss
                        elif self.cooldown_bars_win > 0:
                            cooldown_until[pos.ticker] = date_idx + self.cooldown_bars_win

            for pos in closed_positions:
                if pos in positions:
                    positions.remove(pos)

            # === 2. 新規エントリー ===
            entry_ok = (
                not is_close_time
                and not is_cutoff
                and self._is_entry_allowed(date)
                and abs(daily_loss) < self.initial_capital * self.max_daily_loss
                and not weekly_halted
                and not monthly_halted
            )
            if entry_ok:
                for ticker, df in signals_dict.items():
                    if date not in df.index:
                        continue
                    if len(positions) >= self.max_positions:
                        break
                    if any(p.ticker == ticker for p in positions):
                        continue
                    if self.cooldown_enabled:
                        if ticker in cooldown_until and date_idx < cooldown_until[ticker]:
                            continue

                    row = df.loc[date]
                    signal = row.get("final_signal", "HOLD")
                    if signal not in ("BUY", "SELL"):
                        continue

                    close = row["close"]

                    if self.trend_filter_enabled:
                        tf_col = f"trend_ema_{self.trend_ema_period}"
                        trend_ema = df[tf_col].loc[date]
                        if pd.isna(trend_ema):
                            continue
                        if signal == "BUY" and close < trend_ema:
                            continue
                        if signal == "SELL" and close > trend_ema:
                            continue

                    # 戦略タグ判定
                    strategy_tag = self._get_strategy_tag(row)

                    # BO戦略: entry_start_time〜min(entry_end_time, time_cutoff) のみエントリー許可
                    if strategy_tag == "breakout":
                        bo_effective_end = min(bo_entry_end_int, bo_cutoff_int)
                        if not (bo_entry_start_int <= bar_time_int <= bo_effective_end):
                            continue

                    atr_col = f"ATRr_{self.atr_period}"
                    atr_val = df[atr_col].loc[date]
                    if pd.isna(atr_val) or atr_val <= 0:
                        continue

                    risk_amount = capital * self.risk_per_trade
                    sl_distance = atr_val * self.sl_atr_mult
                    size = (int(risk_amount / sl_distance) // 100) * 100
                    if size < 100:
                        continue

                    position_value = close * size

                    # 戦略別資金上限チェック
                    strategy_exposure = sum(
                        p.entry_price * p.size for p in positions
                        if p.strategy_tag == strategy_tag
                    )
                    if strategy_tag == "mean_reversion":
                        cap_limit = self.mr_capital_limit
                    elif strategy_tag == "breakout":
                        cap_limit = self.bo_capital_limit
                    else:
                        cap_limit = self.initial_capital
                    if strategy_exposure + position_value > cap_limit:
                        remaining_cap = cap_limit - strategy_exposure
                        if remaining_cap < close * 100:
                            continue
                        size = (int(remaining_cap / close) // 100) * 100
                        if size < 100:
                            continue
                        position_value = close * size

                    total_exposure = sum(
                        p.entry_price * p.size for p in positions
                    ) + position_value
                    if total_exposure > self.initial_capital:
                        continue

                    # per-position / total exposure キャップ（live_risk_capsが設定されている場合）
                    if self.max_notional_per_position is not None:
                        effective_per_pos = self.max_notional_per_position * self.safety_margin_ratio
                        if position_value > effective_per_pos:
                            max_size_by_cap = (int(effective_per_pos / close) // 100) * 100
                            if max_size_by_cap < 100:
                                continue
                            size = max_size_by_cap
                            position_value = close * size

                    if self.max_total_exposure is not None:
                        effective_total = self.max_total_exposure * self.safety_margin_ratio
                        current_exposure = sum(p.entry_price * p.size for p in positions)
                        if current_exposure + position_value > effective_total:
                            remaining = effective_total - current_exposure
                            max_size_by_total = (int(remaining / close) // 100) * 100
                            if max_size_by_total < 100:
                                continue
                            size = max_size_by_total
                            position_value = close * size

                    if signal == "BUY":
                        entry_price = close * (1 + self.slippage_rate)
                        stop_loss = entry_price - sl_distance
                        take_profit = entry_price + sl_distance * self.tp_rr_ratio
                        trailing = stop_loss
                        side = Side.LONG
                    else:
                        entry_price = close * (1 - self.slippage_rate)
                        stop_loss = entry_price + sl_distance
                        take_profit = entry_price - sl_distance * self.tp_rr_ratio
                        trailing = stop_loss
                        side = Side.SHORT

                    capital -= abs(entry_price * size * self.commission_rate)

                    # BO用: ORBレンジ取得
                    orb_high = float(row.get("Breakout_orb_high", float("nan")) or float("nan"))
                    orb_low = float(row.get("Breakout_orb_low", float("nan")) or float("nan"))
                    # orb_high/orb_lowがdfに直接あればそちらを使う
                    if math.isnan(orb_high) and "orb_high" in df.columns:
                        v = df["orb_high"].loc[date]
                        orb_high = float(v) if not pd.isna(v) else float("nan")
                    if math.isnan(orb_low) and "orb_low" in df.columns:
                        v = df["orb_low"].loc[date]
                        orb_low = float(v) if not pd.isna(v) else float("nan")

                    partial_targets_copy = list(bo_partial_targets) if strategy_tag == "breakout" else []

                    pos = Position(
                        ticker=ticker, side=side,
                        entry_price=entry_price, entry_date=date,
                        size=size, stop_loss=stop_loss,
                        take_profit=take_profit, trailing_stop=trailing,
                        reason=f"score={row.get('ensemble_score', '')}",
                        strategy_tag=strategy_tag,
                        original_size=size,
                        partial_targets=partial_targets_copy,
                        orb_high=orb_high,
                        orb_low=orb_low,
                    )
                    positions.append(pos)

            # === 3. 資産評価 ===
            unrealized = 0
            for pos in positions:
                if pos.ticker in signals_dict and date in signals_dict[pos.ticker].index:
                    current = signals_dict[pos.ticker].loc[date]["close"]
                    if pos.side == Side.LONG:
                        unrealized += (current - pos.entry_price) * pos.size
                    else:
                        unrealized += (pos.entry_price - current) * pos.size
            equity_curve.append(capital + unrealized)
            equity_dates.append(date)

        if positions:
            for pos in positions:
                if pos.ticker in signals_dict:
                    df = signals_dict[pos.ticker]
                    last_price = df["close"].iloc[-1]
                    trade = self._close_position(
                        pos, last_price, df.index[-1], "BT終了決済",
                    )
                    trades.append(trade)
                    capital += trade.pnl
            positions.clear()

        return BacktestResult(trades=trades, equity_curve=equity_curve, dates=equity_dates)
