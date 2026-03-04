import pandas as pd
import pandas_ta as ta
from strategy.base import StrategyBase, Signal

class TrendFollow(StrategyBase):
    def __init__(self, params: dict):
        super().__init__("TrendFollow", params)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        df = df.copy()
        df["ema_s"] = ta.ema(df["close"], length=p.get("ema_short", 5))
        df["ema_l"] = ta.ema(df["close"], length=p.get("ema_long", 20))
        signals = []
        for i in range(len(df)):
            if i < 1 or pd.isna(df["ema_l"].iloc[i]):
                signals.append(Signal(0, ""))
                continue
            if df["ema_s"].iloc[i] > df["ema_l"].iloc[i] and df["ema_s"].iloc[i-1] <= df["ema_l"].iloc[i-1]:
                signals.append(Signal(1.0, "GC"))
            else:
                signals.append(Signal(0, ""))
        return pd.Series(signals, index=df.index)