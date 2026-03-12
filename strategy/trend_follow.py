"""トレンドフォロー v14: GC/DC 双方向 + MACD確認"""

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

        # MACD
        macd = ta.macd(df["close"],
                       fast=p.get("macd_fast", 12),
                       slow=p.get("macd_slow", 26),
                       signal=p.get("macd_signal", 9))
        if macd is not None and len(macd.columns) >= 3:
            df["macd_line"] = macd.iloc[:, 0]
            df["macd_signal_line"] = macd.iloc[:, 2]
        else:
            df["macd_line"] = 0
            df["macd_signal_line"] = 0

        sell_enabled = p.get("sell_enabled", False)

        signals = []
        for i in range(len(df)):
            if i < 1 or pd.isna(df["ema_l"].iloc[i]):
                signals.append(Signal(0, ""))
                continue

            score = 0.0
            reasons = []

            # ゴールデンクロス
            if (df["ema_s"].iloc[i] > df["ema_l"].iloc[i]
                    and df["ema_s"].iloc[i-1] <= df["ema_l"].iloc[i-1]):
                score += 0.7
                reasons.append("GC")

                # MACD確認
                if df["macd_line"].iloc[i] > df["macd_signal_line"].iloc[i]:
                    score += 0.3
                    reasons.append("MACD+")

            # デッドクロス（売り）
            if sell_enabled:
                if (df["ema_s"].iloc[i] < df["ema_l"].iloc[i]
                        and df["ema_s"].iloc[i-1] >= df["ema_l"].iloc[i-1]):
                    score -= 0.7
                    reasons.append("DC")

                    if df["macd_line"].iloc[i] < df["macd_signal_line"].iloc[i]:
                        score -= 0.3
                        reasons.append("MACD-")

            signals.append(Signal(score, " ".join(reasons)))

        return pd.Series(signals, index=df.index)
