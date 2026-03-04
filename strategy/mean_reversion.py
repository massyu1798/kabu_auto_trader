import pandas as pd
import pandas_ta as ta
from strategy.base import StrategyBase, Signal

class MeanReversion(StrategyBase):
    def __init__(self, params: dict):
        super().__init__("MeanReversion", params)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        df = df.copy()
        df["rsi"] = ta.rsi(df["close"], length=p.get("rsi_period", 14))
        signals = []
        for i in range(len(df)):
            if pd.isna(df["rsi"].iloc[i]):
                signals.append(Signal(0, ""))
                continue
            if df["rsi"].iloc[i] < p.get("rsi_oversold", 25):
                signals.append(Signal(1.0, "Oversold"))
            else:
                signals.append(Signal(0, ""))
        return pd.Series(signals, index=df.index)