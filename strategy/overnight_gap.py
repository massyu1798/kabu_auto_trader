"""戦略③ オーバーナイト・ギャップ戦略 v14
- IBS (Internal Bar Strength) + RSI(2) の極端な売られすぎで引け買い
- 翌営業日の寄り付きで決済
- main_backtest_combined.py から日足ベースで呼び出す想定
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from strategy.base import StrategyBase, Signal


class OvernightGap(StrategyBase):
    def __init__(self, params: dict):
        super().__init__("OvernightGap", params)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        df = df.copy()

        # IBS = (Close - Low) / (High - Low)
        hl_range = df["high"] - df["low"]
        df["ibs"] = (df["close"] - df["low"]) / hl_range.replace(0, np.nan)

        # RSI(2) - 超短期RSI
        df["rsi2"] = ta.rsi(df["close"], length=p.get("rsi_short_period", 2))

        # 当日リターン
        df["daily_return"] = df["close"].pct_change() * 100

        ibs_oversold = p.get("ibs_oversold", 0.2)
        ibs_overbought = p.get("ibs_overbought", 0.8)
        rsi2_oversold = p.get("rsi2_oversold", 10)
        rsi2_overbought = p.get("rsi2_overbought", 90)
        min_drop = p.get("min_daily_drop_pct", -1.5)
        min_rise = p.get("min_daily_rise_pct", 1.5)

        signals = []
        for i in range(len(df)):
            ibs = df["ibs"].iloc[i]
            rsi2 = df["rsi2"].iloc[i]
            daily_ret = df["daily_return"].iloc[i]

            if pd.isna(ibs) or pd.isna(rsi2) or pd.isna(daily_ret):
                signals.append(Signal(0, ""))
                continue

            score = 0.0
            reasons = []

            # === 引け買いシグナル ===
            if ibs < ibs_oversold:
                score += 0.5
                reasons.append(f"IBS={ibs:.2f}")
            if rsi2 < rsi2_oversold:
                score += 0.4
                reasons.append(f"RSI2={rsi2:.0f}")
            if daily_ret <= min_drop:
                score += 0.3
                reasons.append(f"Drop={daily_ret:.1f}%")

            # === 引け売りシグナル ===
            if ibs > ibs_overbought:
                score -= 0.5
                reasons.append(f"IBS={ibs:.2f}")
            if rsi2 > rsi2_overbought:
                score -= 0.4
                reasons.append(f"RSI2={rsi2:.0f}")
            if daily_ret >= min_rise:
                score -= 0.3
                reasons.append(f"Rise={daily_ret:.1f}%")

            # 金曜日フィルター（週末リスク回避）
            if hasattr(df.index[i], "weekday") and df.index[i].weekday() == 4:
                score *= 0.3  # 金曜は大幅に減衰
                reasons.append("FriDiscount")

            signals.append(Signal(score, " ".join(reasons)))

        return pd.Series(signals, index=df.index)
