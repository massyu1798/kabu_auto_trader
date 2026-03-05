"""
5-minute OHLCV bar builder for live trading (v13)

Builds 5-minute bars from /board API responses.
TradingVolume from /board is cumulative, so we take diffs for bar volume.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class BarBuilder:
    """
    Build 5-minute OHLCV bars per ticker from live /board data.

    Usage:
        builder = BarBuilder()
        # on each /board poll:
        builder.update(ticker, board_data)
        # get bars DataFrame:
        df = builder.get_bars(ticker)   # columns: open, high, low, close, volume
    """

    def __init__(self, bar_interval_min: int = 5):
        self.bar_interval_min = bar_interval_min
        # Per-ticker state
        # {ticker: {"bars": [...], "current_bar": {...}, "prev_cum_volume": float}}
        self._state: dict[str, dict] = {}

    def _get_bar_start(self, dt: datetime) -> datetime:
        """Round down to the nearest bar boundary."""
        minute = (dt.minute // self.bar_interval_min) * self.bar_interval_min
        return dt.replace(minute=minute, second=0, microsecond=0)

    def _ensure_ticker(self, ticker: str):
        if ticker not in self._state:
            self._state[ticker] = {
                "bars": [],           # list of completed bar dicts
                "current_bar": None,  # bar being built
                "prev_cum_volume": None,
            }

    def update(self, ticker: str, board: dict) -> None:
        """
        Feed a /board response for a ticker.

        Expected board keys:
            CurrentPrice, HighPrice, LowPrice, OpeningPrice, TradingVolume, VWAP
        """
        if not board:
            return
        current_price = board.get("CurrentPrice")
        if current_price is None:
            return

        current_price = float(current_price)
        high_price = float(board.get("HighPrice", current_price))
        low_price = float(board.get("LowPrice", current_price))
        # /board OpeningPrice = today's opening price (use current as bar-open fallback)
        cum_volume = float(board.get("TradingVolume", 0))
        vwap = board.get("VWAP")

        now = datetime.now()
        bar_start = self._get_bar_start(now)

        self._ensure_ticker(ticker)
        state = self._state[ticker]

        # Calculate incremental volume from cumulative TradingVolume
        tick_volume = 0.0
        if state["prev_cum_volume"] is not None:
            diff = cum_volume - state["prev_cum_volume"]
            tick_volume = max(diff, 0)  # guard against reset
        state["prev_cum_volume"] = cum_volume

        # Check if we need to close current bar and start a new one
        if state["current_bar"] is not None:
            if bar_start > state["current_bar"]["bar_start"]:
                # Finalize current bar
                self._close_current_bar(ticker)

        # If no current bar, start a new one
        if state["current_bar"] is None:
            state["current_bar"] = {
                "bar_start": bar_start,
                "open": current_price,
                "high": current_price,
                "low": current_price,
                "close": current_price,
                "volume": tick_volume,
            }
        else:
            # Update current bar
            bar = state["current_bar"]
            bar["high"] = max(bar["high"], current_price)
            bar["low"] = min(bar["low"], current_price)
            bar["close"] = current_price
            bar["volume"] += tick_volume

        # Also store VWAP for reference
        if vwap is not None:
            state["current_bar"]["vwap"] = float(vwap)

    def _close_current_bar(self, ticker: str):
        """Finalize current bar and move to completed bars."""
        state = self._state[ticker]
        bar = state["current_bar"]
        if bar is None:
            return
        state["bars"].append(bar)
        state["current_bar"] = None

        # Keep max 500 bars (~41 hours) to avoid memory bloat
        if len(state["bars"]) > 500:
            state["bars"] = state["bars"][-500:]

    def get_bars(self, ticker: str, include_current: bool = True) -> pd.DataFrame:
        """
        Return completed bars as DataFrame.
        If include_current=True, the in-progress bar is appended.

        Returns DataFrame with columns: open, high, low, close, volume
        Index: DatetimeIndex from bar_start
        """
        self._ensure_ticker(ticker)
        state = self._state[ticker]

        bars = list(state["bars"])
        if include_current and state["current_bar"] is not None:
            bars.append(state["current_bar"])

        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(bars)
        df.index = pd.to_datetime(df["bar_start"])
        df = df[["open", "high", "low", "close", "volume"]]
        return df

    def get_bar_count(self, ticker: str) -> int:
        """Return number of completed bars for a ticker."""
        self._ensure_ticker(ticker)
        state = self._state[ticker]
        n = len(state["bars"])
        if state["current_bar"] is not None:
            n += 1
        return n

    def reset(self, ticker: str = None):
        """Reset state for one or all tickers."""
        if ticker:
            self._state.pop(ticker, None)
        else:
            self._state.clear()
