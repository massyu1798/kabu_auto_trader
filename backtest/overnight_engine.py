"""
オーバーナイト・ギャップ (ONG) バックテストエンジン v1.0

ロジック:
  - シグナル当日の引け値（スリッページ込み）でロングエントリー
  - 翌営業日の寄り付きで成行決済
  - ギャップアップ → 寄り付きで利確
  - ギャップダウン → -stop_loss_pct% の損切りキャップ（下限）
  - ATR(14) ベースのポジションサイジング:
      size = max_risk_per_trade / (ATR(14) × atr_risk_multiplier)
"""

from dataclasses import dataclass, field

import pandas as pd
import pandas_ta as ta
import yaml


@dataclass
class Trade:
    ticker: str
    side: str
    entry_price: float
    exit_price: float
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    size: int
    pnl: float
    pnl_pct: float
    entry_reason: str = "ONG_entry"
    exit_reason: str = ""


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    dates: list = field(default_factory=list)


class OvernightGapEngine:
    def __init__(self, config_path: str = "config/overnight_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        g = self.config.get("global", {})
        self.initial_capital = float(g.get("initial_capital", 3_000_000))
        self.max_risk_per_trade = float(g.get("max_risk_per_trade", 100_000))
        self.slippage_rate = float(g.get("slippage_rate", 0.001))
        self.commission_rate = float(g.get("commission_rate", 0.0))

        ong = self.config.get("ong", {})
        self.atr_period = int(ong.get("atr_period", 14))
        self.atr_risk_multiplier = float(ong.get("atr_risk_multiplier", 2.0))
        # stop_loss_pct は負の値 (例: -1.0 → -1.0%)
        self.stop_loss_pct = float(ong.get("stop_loss_pct", -1.0))

    # ─────────────────────────────────────────────────────────────
    # メイン: バックテスト実行
    # ─────────────────────────────────────────────────────────────
    def run(self, signals_dict: dict) -> BacktestResult:
        """
        Parameters
        ----------
        signals_dict : dict[str, pd.DataFrame]
            generate_ong_signals() が返す {ticker: daily_df with ONG_signal} 辞書

        Returns
        -------
        BacktestResult
            trades, equity_curve (資産推移), dates
        """
        atr_col = f"ATR_{self.atr_period}"
        all_trades: list[Trade] = []

        for ticker, df in signals_dict.items():
            if df is None or df.empty:
                continue

            d = df.copy()

            # ATR 計算（未計算の場合）
            if atr_col not in d.columns:
                d[atr_col] = ta.atr(d["high"], d["low"], d["close"],
                                    length=self.atr_period)

            for i in range(len(d) - 1):
                row = d.iloc[i]

                if not row.get("ONG_signal", False):
                    continue

                atr_val = row.get(atr_col, float("nan"))
                if pd.isna(atr_val) or atr_val <= 0:
                    continue

                # ── ポジションサイジング ─────────────────────────
                risk_distance = atr_val * self.atr_risk_multiplier
                size = int(self.max_risk_per_trade / risk_distance)
                size = (size // 100) * 100   # 100株単位に丸め
                if size <= 0:
                    continue

                # ── エントリー価格（引け + スリッページ）─────────
                entry_price = float(row["close"]) * (1 + self.slippage_rate)

                # ── 翌日寄り付き ──────────────────────────────────
                next_row = d.iloc[i + 1]
                next_open = float(next_row["open"])

                # ── エグジット価格決定 ─────────────────────────────
                # ギャップダウン時: -stop_loss_pct% が下限
                sl_floor = entry_price * (1 + self.stop_loss_pct / 100.0)

                if next_open >= entry_price:
                    # ギャップアップ or 変わらず → 寄り付き利確
                    raw_exit = next_open
                    exit_reason = "翌寄り利確" if next_open > entry_price else "翌寄り決済(同値)"
                else:
                    # ギャップダウン → 損切りキャップ
                    raw_exit = max(next_open, sl_floor)
                    if next_open < sl_floor:
                        exit_reason = f"翌寄り損切り({self.stop_loss_pct:.1f}%キャップ)"
                    else:
                        exit_reason = "翌寄り決済(小GD)"

                exit_price = raw_exit * (1 - self.slippage_rate)

                # ── PnL 計算 ─────────────────────────────────────
                gross_pnl = (exit_price - entry_price) * size
                commission = (
                    abs(entry_price * size * self.commission_rate)
                    + abs(exit_price * size * self.commission_rate)
                )
                pnl = gross_pnl - commission
                pnl_pct = pnl / (entry_price * size) * 100 if entry_price * size != 0 else 0.0

                all_trades.append(Trade(
                    ticker=ticker,
                    side="LONG",
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_date=d.index[i],
                    exit_date=d.index[i + 1],
                    size=size,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    entry_reason="ONG_entry",
                    exit_reason=exit_reason,
                ))

        # ── 日付順にソートして資産曲線を構築 ────────────────────
        all_trades.sort(key=lambda t: t.entry_date)

        capital = self.initial_capital
        equity_curve: list[float] = []
        equity_dates: list = []

        for trade in all_trades:
            capital += trade.pnl
            equity_curve.append(capital)
            equity_dates.append(trade.exit_date)

        return BacktestResult(
            trades=all_trades,
            equity_curve=equity_curve,
            dates=equity_dates,
        )
