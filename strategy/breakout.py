"""戦略② ORBブレイクアウト・モメンタム v14
- Opening Range Breakout (寄り30分レンジ)
- チャネルブレイクアウト（従来版）との併用
- MACD確認フィルター付き
"""

import pandas as pd
import pandas_ta as ta
from strategy.base import StrategyBase, Signal


class Breakout(StrategyBase):
    def __init__(self, params: dict):
        super().__init__("Breakout", params)

    def _calc_orb(self, df: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
        """各日の寄り付きN分間のレンジ（高値・安値）を算出"""
        orb_high = pd.Series(index=df.index, dtype=float, data=float("nan"))
        orb_low = pd.Series(index=df.index, dtype=float, data=float("nan"))

        for day, group in df.groupby(df.index.date):
            first_time = group.index[0]
            cutoff = first_time + pd.Timedelta(minutes=minutes)
            opening_range = group[group.index <= cutoff]

            if len(opening_range) == 0:
                continue

            day_orb_high = opening_range["high"].max()
            day_orb_low = opening_range["low"].min()

            orb_high.loc[group.index] = day_orb_high
            orb_low.loc[group.index] = day_orb_low

        return pd.DataFrame({"orb_high": orb_high, "orb_low": orb_low}, index=df.index)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        df = df.copy()

        use_orb = p.get("use_orb", False)
        macd_confirm = p.get("macd_confirm", False)

        # チャネルブレイクアウト（従来）
        period = p.get("channel_period", 20)
        df["high_max"] = df["high"].rolling(window=period).max().shift(1)
        df["low_min"] = df["low"].rolling(window=period).min().shift(1)

        # ORB
        if use_orb:
            orb_df = self._calc_orb(df, minutes=p.get("orb_minutes", 30))
            df["orb_high"] = orb_df["orb_high"]
            df["orb_low"] = orb_df["orb_low"]

        # MACD
        if macd_confirm:
            macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
            if macd is not None and len(macd.columns) >= 3:
                df["macd_line"] = macd.iloc[:, 0]
                df["macd_signal"] = macd.iloc[:, 2]
            else:
                df["macd_line"] = 0
                df["macd_signal"] = 0

        # 出来高トレンド
        df["vol_ma5"] = df["volume"].rolling(5).mean()

        signals = []
        for i in range(len(df)):
            score = 0.0
            reasons = []

            close = df["close"].iloc[i]

            # --- チャネルブレイクアウト（上方向） ---
            if not pd.isna(df["high_max"].iloc[i]) and close > df["high_max"].iloc[i]:
                score += 0.6
                reasons.append("ChBreakout")

            # --- チャネルブレイクダウン（下方向） ---
            if not pd.isna(df["low_min"].iloc[i]) and close < df["low_min"].iloc[i]:
                score -= 0.6
                reasons.append("ChBreakdown")

            # --- ORBブレイクアウト ---
            if use_orb and "orb_high" in df.columns:
                orb_h = df["orb_high"].iloc[i]
                orb_l = df["orb_low"].iloc[i]
                if not pd.isna(orb_h) and close > orb_h:
                    score += 0.5
                    reasons.append("ORB_Up")
                elif not pd.isna(orb_l) and close < orb_l:
                    score -= 0.5
                    reasons.append("ORB_Down")

            # --- MACD確認 ---
            if macd_confirm and "macd_line" in df.columns:
                ml = df["macd_line"].iloc[i]
                ms = df["macd_signal"].iloc[i]
                if not pd.isna(ml) and not pd.isna(ms):
                    if score > 0 and ml > ms:
                        score += 0.3
                        reasons.append("MACD_Bull")
                    elif score < 0 and ml < ms:
                        score -= 0.3
                        reasons.append("MACD_Bear")

            # --- 出来高増加トレンド ---
            if not pd.isna(df["vol_ma5"].iloc[i]) and df["vol_ma5"].iloc[i] > 0:
                if df["volume"].iloc[i] > df["vol_ma5"].iloc[i] * 1.3:
                    if score > 0:
                        score += 0.2
                    elif score < 0:
                        score -= 0.2
                    reasons.append("VolUp")

            signals.append(Signal(score, " ".join(reasons)))

        return pd.Series(signals, index=df.index)
