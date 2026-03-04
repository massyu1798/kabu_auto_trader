import pandas as pd
from strategy.base import StrategyBase, Signal

class Breakout(StrategyBase):
    def __init__(self, params: dict):
        super().__init__("Breakout", params)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        df = df.copy()
        period = p.get("channel_period", 20)
        df["high_max"] = df["high"].rolling(window=period).max().shift(1)
        signals = []
        for i in range(len(df)):
            if pd.isna(df["high_max"].iloc[i]):
                signals.append(Signal(0, ""))
                continue
            if df["close"].iloc[i] > df["high_max"].iloc[i]:
                signals.append(Signal(1.0, "Breakout"))
            else:
                signals.append(Signal(0, ""))
        return pd.Series(signals, index=df.index)