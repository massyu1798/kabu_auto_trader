"""午後リバーサル専用バックテストエンジン"""

from dataclasses import dataclass, field
from enum import Enum
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
    vwap_target: float = 0.0      # VWAP回帰目標
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
    def __init__(self, config_path: str = "config/afternoon_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        g = self.config["global"]
        self.initial_capital = g["initial_capital"]
        self.max_positions = g["max_positions"]
        self.risk_per_trade = g["risk_per_trade"]
        self.max_daily_loss = g["max_daily_loss"]
        self.commission_rate = g["commission_rate"]
        self.slippage_rate = g["slippage_rate"]

        e = self.config["exit"]
        self.sl_atr_mult = e["stop_loss_atr_multiplier"]
        self.tp_atr_mult = e.get("take_profit_atr_multiplier", 2.0)
        self.atr_period = e.get("atr_period", 14)
        self.trailing_enabled = e.get("trailing_stop", False)
        self.trailing_atr_mult = e.get("trailing_atr_multiplier", 1.5)
        self.force_close_time = e.get("force_close_time", 1450)

        cd = self.config.get("cooldown", {})
        self.cooldown_enabled = cd.get("enabled", False)
        self.cooldown_bars_loss = cd.get("bars_after_loss", 6)
        self.cooldown_bars_win = cd.get("bars_after_win", 0)

    def _get_time_int(self, timestamp):
        if not hasattr(timestamp, "hour"):
            return 0
        return timestamp.hour * 100 + timestamp.minute

    def _is_force_close(self, timestamp):
        return self._get_time_int(timestamp) >= self.force_close_time

    def _is_market_close(self, timestamp):
        return self._get_time_int(timestamp) >= 1520

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

        # 前処理: ATR算出
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

        daily_loss = 0.0
        current_day = None
        cooldown_until = {}

        for date_idx, date in enumerate(all_dates):
            day = date.date() if hasattr(date, "date") else date

            # 日替わり処理
            if current_day != day:
                # 前日の��存ポジション強制決済
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
                daily_loss = 0.0
                current_day = day
                cooldown_until.clear()

            is_force_close = self._is_force_close(date)
            is_market_close = self._is_market_close(date)

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

                # トレーリングストップ更新
                if self.trailing_enabled:
                    atr_col = f"ATRr_{self.atr_period}"
                    atr_val = df[atr_col].loc[date]
                    if not pd.isna(atr_val):
                        if pos.side == Side.LONG:
                            new_trail = high - atr_val * self.trailing_atr_mult
                            if new_trail > pos.trailing_stop:
                                pos.trailing_stop = new_trail
                        else:
                            new_trail = low + atr_val * self.trailing_atr_mult
                            if new_trail < pos.trailing_stop:
                                pos.trailing_stop = new_trail

                # エグジット判定
                if is_force_close or is_market_close:
                    exit_reason = f"時間決済 ({self._get_time_int(date)})"
                elif pos.side == Side.LONG and current_price <= pos.stop_loss:
                    exit_reason = f"損切り ({pos.stop_loss:.0f})"
                elif pos.side == Side.SHORT and current_price >= pos.stop_loss:
                    exit_reason = f"損切り ({pos.stop_loss:.0f})"
                # VWAP回帰利確（リバーサルの主要エグジット）
                elif pos.side == Side.LONG and pos.vwap_target > 0 and current_price >= pos.vwap_target:
                    exit_reason = f"VWAP回帰利確 ({pos.vwap_target:.0f})"
                elif pos.side == Side.SHORT and pos.vwap_target > 0 and current_price <= pos.vwap_target:
                    exit_reason = f"VWAP回帰利確 ({pos.vwap_target:.0f})"
                # ATRベー���利確（VWAPが遠い場合のバックアップ）
                elif pos.side == Side.LONG and current_price >= pos.take_profit:
                    exit_reason = f"利確 ({pos.take_profit:.0f})"
                elif pos.side == Side.SHORT and current_price <= pos.take_profit:
                    exit_reason = f"利確 ({pos.take_profit:.0f})"
                elif self.trailing_enabled:
                    if pos.side == Side.LONG and current_price <= pos.trailing_stop:
                        exit_reason = f"TS ({pos.trailing_stop:.0f})"
                    elif pos.side == Side.SHORT and current_price >= pos.trailing_stop:
                        exit_reason = f"TS ({pos.trailing_stop:.0f})"

                if exit_reason:
                    trade = self._close_position(pos, current_price, date, exit_reason)
                    trades.append(trade)
                    capital += trade.pnl
                    daily_loss += min(0, trade.pnl)
                    closed_positions.append(pos)

                    if self.cooldown_enabled:
                        if trade.pnl <= 0:
                            cooldown_until[pos.ticker] = date_idx + self.cooldown_bars_loss
                        elif self.cooldown_bars_win > 0:
                            cooldown_until[pos.ticker] = date_idx + self.cooldown_bars_win

            for pos in closed_positions:
                positions.remove(pos)

            # === 2. 新規エントリー ===
            if (
                not is_force_close
                and not is_market_close
                and abs(daily_loss) < self.initial_capital * self.max_daily_loss
            ):
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
                    signal = row.get("afternoon_signal", "HOLD")
                    if signal not in ("BUY", "SELL"):
                        continue

                    close = row["close"]
                    vwap = row.get("vwap", 0)

                    atr_col = f"ATRr_{self.atr_period}"
                    atr_val = df[atr_col].loc[date]
                    if pd.isna(atr_val) or atr_val <= 0:
                        continue

                    risk_amount = capital * self.risk_per_trade
                    sl_distance = atr_val * self.sl_atr_mult
                    size = int(risk_amount / sl_distance)
                    if size <= 0:
                        continue

                    position_value = close * size
                    total_exposure = sum(
                        p.entry_price * p.size for p in positions
                    ) + position_value
                    if total_exposure > self.initial_capital:
                        continue

                    if signal == "BUY":
                        entry_price = close * (1 + self.slippage_rate)
                        stop_loss = entry_price - sl_distance
                        take_profit = entry_price + atr_val * self.tp_atr_mult
                        trailing = stop_loss
                        vwap_target = vwap if vwap > entry_price else take_profit
                        side = Side.LONG
                    else:
                        entry_price = close * (1 - self.slippage_rate)
                        stop_loss = entry_price + sl_distance
                        take_profit = entry_price - atr_val * self.tp_atr_mult
                        trailing = stop_loss
                        vwap_target = vwap if vwap < entry_price else take_profit
                        side = Side.SHORT

                    capital -= abs(entry_price * size * self.commission_rate)

                    pos = Position(
                        ticker=ticker, side=side,
                        entry_price=entry_price, entry_date=date,
                        size=size, stop_loss=stop_loss,
                        take_profit=take_profit, trailing_stop=trailing,
                        vwap_target=vwap_target,
                        reason=f"reversal morning_move={row.get('morning_move', 0):.1f}%",
                    )
                    positions.append(pos)

            # === 3. ���産評価 ===
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

        # 残存ポジション決済
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
