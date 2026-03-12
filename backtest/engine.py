"""バックテストエンジン v12.5: 9:05〜14:00エントリー許可 + max_positions=8"""

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


class BacktestEngine:
    def __init__(self, config_path: str = "config/strategy_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        g = self.config["global"]
        self.initial_capital = g["initial_capital"]
        self.max_positions = g["max_positions"]
        self.risk_per_trade = g["risk_per_trade"]
        self.max_daily_loss = g["max_daily_loss"]
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
        """9:05〜14:00 をエントリー許可時間帯に変更（後場前半まで許可）"""
        if not hasattr(timestamp, "hour"):
            return True
        t = timestamp.hour * 100 + timestamp.minute
        return 905 <= t <= 1400

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
        current_day = None
        is_daytrade = self.max_holding_days == 0
        cooldown_until = {}
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        for date_idx, date in enumerate(all_dates):
            day = date.date() if hasattr(date, "date") else date
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
                    positions.clear()
                daily_loss = 0.0
                current_day = day
                cooldown_until.clear()

            is_close_time = self._is_market_close(date)
            is_cutoff = self._is_entry_cutoff(date)

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

                # トレーリングストップ更新（v12.4: 加速TS）
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

                if is_daytrade and is_close_time:
                    exit_reason = "引け強制決"
                elif pos.side == Side.LONG and current_price <= pos.stop_loss:
                    exit_reason = f"損切り ({pos.stop_loss:.0f})"
                elif pos.side == Side.SHORT and current_price >= pos.stop_loss:
                    exit_reason = f"損切り ({pos.stop_loss:.0f})"
                elif pos.side == Side.LONG and current_price >= pos.take_profit:
                    exit_reason = f"利�� ({pos.take_profit:.0f})"
                elif pos.side == Side.SHORT and current_price <= pos.take_profit:
                    exit_reason = f"利確 ({pos.take_profit:.0f})"
                elif self.trailing_enabled:
                    if pos.side == Side.LONG and current_price <= pos.trailing_stop:
                        exit_reason = f"TS ({pos.trailing_stop:.0f})"
                    elif pos.side == Side.SHORT and current_price >= pos.trailing_stop:
                        exit_reason = f"TS ({pos.trailing_stop:.0f})"

                if exit_reason is None:
                    if pos.side == Side.LONG and row.get("final_signal") == "SELL":
                        exit_reason = "反対シグナル"
                    elif pos.side == Side.SHORT and row.get("final_signal") == "BUY":
                        exit_reason = "反対シグナル"

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
                not is_close_time
                and not is_cutoff
                and self._is_entry_allowed(date)
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

                    pos = Position(
                        ticker=ticker, side=side,
                        entry_price=entry_price, entry_date=date,
                        size=size, stop_loss=stop_loss,
                        take_profit=take_profit, trailing_stop=trailing,
                        reason=f"score={row.get('ensemble_score', '')}",
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
