"""
バックテストエンジン: 午後リバーサル専用
- エン���リー: 12:30���14:00
- エグジット: VWAP回帰 or SL or 14:50強制決済
- 空売り対応
"""

from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import pandas_ta as ta
import numpy as np
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
    vwap_target: float = 0.0
    reason: str = ""


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


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    dates: list = field(default_factory=list)


class AfternoonBacktestEngine:
    def __init__(self, strategy_config_path: str = "config/strategy_config.yaml",
                 afternoon_config_path: str = "config/afternoon_config.yaml"):
        with open(strategy_config_path, "r", encoding="utf-8") as f:
            self.strategy_config = yaml.safe_load(f)
        with open(afternoon_config_path, "r", encoding="utf-8") as f:
            self.afternoon_config = yaml.safe_load(f)

        g = self.strategy_config["global"]
        self.initial_capital = g["initial_capital"]
        self.commission_rate = g["commission_rate"]
        self.slippage_rate = g["slippage_rate"]

        a = self.afternoon_config["afternoon_reversal"]
        self.max_positions = a.get("max_positions", 3)
        self.risk_per_trade = a.get("risk_per_trade", 0.005)
        self.sl_atr_multiplier = a.get("sl_atr_multiplier", 1.5)
        self.vwap_tp_ratio = a.get("vwap_tp_ratio", 0.7)
        self.force_close_time = a.get("force_close_time", 1450)
        self.cooldown_bars = a.get("cooldown_bars_after_loss", 6)
        self.atr_period = a.get("atr_period", 14)

    def _is_force_close(self, timestamp):
        if not hasattr(timestamp, "hour"):
            return False
        t = timestamp.hour * 100 + timestamp.minute
        return t >= self.force_close_time

    def _close_position(self, pos, current_price, date, reason):
        if pos.side == Side.LONG:
            exit_price = current_price * (1 - self.slippage_rate)
            pnl = (exit_price - pos.entry_price) * pos.size
        else:
            exit_price = current_price * (1 + self.slippage_rate)
            pnl = (pos.entry_price - exit_price) * pos.size
        pnl -= abs(exit_price * pos.size * self.commission_rate)
        pnl_pct = pnl / (pos.entry_price * pos.size) * 100
        return Trade(
            ticker=pos.ticker, side=pos.side,
            entry_price=pos.entry_price, exit_price=exit_price,
            entry_date=pos.entry_date, exit_date=date,
            size=pos.size, pnl=pnl, pnl_pct=pnl_pct,
            entry_reason=pos.reason, exit_reason=reason,
        )

    def run(self, signals_dict: dict[str, pd.DataFrame]) -> BacktestResult:
        capital = self.initial_capital
        positions: list[Position] = []
        trades: list[Trade] = []
        equity_curve = []
        equity_dates = []

        # ATR計算
        for ticker, df in signals_dict.items():
            atr_col = f"ATRr_{self.atr_period}"
            if atr_col not in df.columns:
                df[atr_col] = ta.atr(
                    df["high"], df["low"], df["close"],
                    length=self.atr_period,
                )

        all_dates = sorted(set(
            date for df in signals_dict.values() for date in df.index
        ))

        current_day = None
        cooldown_until = {}

        for date_idx, date in enumerate(all_dates):
            day = date.date() if hasattr(date, "date") else date

            # 日替わり: ポジションクリア
            if current_day != day:
                if positions:
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
                    positions.clear()
                current_day = day
                cooldown_until.clear()

            is_force_close = self._is_force_close(date)

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
                exit_reason = None

                # 14:50 強制決済
                if is_force_close:
                    exit_reason = "午後強制決済"

                # 損切り
                elif pos.side == Side.LONG and current_price <= pos.stop_loss:
                    exit_reason = f"損切り ({pos.stop_loss:.0f})"
                elif pos.side == Side.SHORT and current_price >= pos.stop_loss:
                    exit_reason = f"損切り ({pos.stop_loss:.0f})"

                # VWAP回帰で利確
                elif pos.side == Side.LONG and current_price >= pos.vwap_target:
                    exit_reason = f"VWAP回帰利確 ({pos.vwap_target:.0f})"
                elif pos.side == Side.SHORT and current_price <= pos.vwap_target:
                    exit_reason = f"VWAP回帰利確 ({pos.vwap_target:.0f})"

                # 固定利確
                elif pos.side == Side.LONG and current_price >= pos.take_profit:
                    exit_reason = f"利確 ({pos.take_profit:.0f})"
                elif pos.side == Side.SHORT and current_price <= pos.take_profit:
                    exit_reason = f"利確 ({pos.take_profit:.0f})"

                if exit_reason:
                    trade = self._close_position(pos, current_price, date, exit_reason)
                    trades.append(trade)
                    capital += trade.pnl
                    closed_positions.append(pos)

                    if trade.pnl <= 0:
                        cooldown_until[pos.ticker] = date_idx + self.cooldown_bars

            for pos in closed_positions:
                positions.remove(pos)

            # === 2. 新規エントリー ===
            if not is_force_close:
                for ticker, df in signals_dict.items():
                    if date not in df.index:
                        continue
                    if len(positions) >= self.max_positions:
                        break
                    if any(p.ticker == ticker for p in positions):
                        continue
                    if ticker in cooldown_until and date_idx < cooldown_until[ticker]:
                        continue

                    row = df.loc[date]
                    signal = row.get("afternoon_signal", "HOLD")
                    if signal not in ("BUY", "SELL"):
                        continue

                    close = row["close"]
                    vwap_val = row.get("vwap", np.nan)
                    atr_col = f"ATRr_{self.atr_period}"
                    atr_val = df[atr_col].loc[date]

                    if pd.isna(atr_val) or atr_val <= 0:
                        continue

                    risk_amount = capital * self.risk_per_trade
                    sl_distance = atr_val * self.sl_atr_multiplier
                    size = int(risk_amount / sl_distance)
                    if size <= 0:
                        continue

                    if signal == "BUY":
                        entry_price = close * (1 + self.slippage_rate)
                        stop_loss = entry_price - sl_distance
                        # VWAP回帰を利確ターゲットに
                        if not pd.isna(vwap_val):
                            vwap_target = entry_price + (vwap_val - entry_price) * self.vwap_tp_ratio
                        else:
                            vwap_target = entry_price + sl_distance * 1.5
                        take_profit = entry_price + sl_distance * 2.5
                        side = Side.LONG
                    else:
                        entry_price = close * (1 - self.slippage_rate)
                        stop_loss = entry_price + sl_distance
                        if not pd.isna(vwap_val):
                            vwap_target = entry_price - (entry_price - vwap_val) * self.vwap_tp_ratio
                        else:
                            vwap_target = entry_price - sl_distance * 1.5
                        take_profit = entry_price - sl_distance * 2.5
                        side = Side.SHORT

                    capital -= abs(entry_price * size * self.commission_rate)

                    pos = Position(
                        ticker=ticker, side=side,
                        entry_price=entry_price, entry_date=date,
                        size=size, stop_loss=stop_loss,
                        take_profit=take_profit,
                        trailing_stop=stop_loss,
                        vwap_target=vwap_target,
                        reason=f"afternoon_reversal",
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

        # 残ポジション決済
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
