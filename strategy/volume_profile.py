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
                # 直近の価格変動方向を考慮してスコアの正負を判定
                if i >= 3:
                    price_change = df["close"].iloc[i] - df["close"].iloc[i - 3]
                else:
                    price_change = 0.0
                threshold = df["close"].iloc[i] * 0.001  # 0.001 (0.1%) を横ばい判定閾値
                if price_change < -threshold:
                    # 下落 + 出来高急増 → セリングクライマックス（反発期待の買い）
                    signals.append(Signal(1.0, "VolSurge_Drop"))
                elif price_change > threshold:
                    # 上昇 + 出来高急増 → 過熱・利確売りシグナル
                    signals.append(Signal(-0.5, "VolSurge_Rise"))
                else:
                    # 横ばい + 出来高急増 → 弱めの買いシグナル
                    signals.append(Signal(0.3, "VolSurge_Flat"))
            else:
                signals.append(Signal(0, ""))
        return pd.Series(signals, index=df.index)
