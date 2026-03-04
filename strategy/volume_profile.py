import pandas as pd
from strategy.base import StrategyBase, Signal

class VolumeProfile(StrategyBase):
    def __init__(self, params: dict):
        super().__init__("VolumeProfile", params)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        df = df.copy()
        period = p.get("vwap_period", 20)
        df["vol_ma"] = df["volume"].rolling(window=period).mean()
        signals = []
        for i in range(len(df)):
            if pd.isna(df["vol_ma"].iloc[i]) or df["vol_ma"].iloc[i] == 0:
                signals.append(Signal(0, ""))
                continue
            if df["volume"].iloc[i] > df["vol_ma"].iloc[i] * p.get("volume_surge_ratio", 1.5):
                signals.append(Signal(1.0, "VolSurge"))
            else:
                signals.append(Signal(0, ""))
        return pd.Series(signals, index=df.index)