"""戦略② ORBブレイクアウト・モメンタム v20
- Opening Range Breakout (寄り30分レンジ)
- チャネルブレイクアウト（従来版）との併用
- 独立判定型: ATR必須 + 出来高必須 + ブレイクアウト成立で最低スコア1.0を保証
- MACD確認ボーナス: +0.5
- チャネル・ORB両方成立ボーナス: +0.5
- エントリー時間帯制限: 9:35〜13:00
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

    def _calc_prev_day_atr(self, df: pd.DataFrame, atr_period: int = 14) -> pd.Series:
        """前日ATRを日次集計して各バーに対応付ける"""
        atr_series = ta.atr(df["high"], df["low"], df["close"], length=atr_period)
        if atr_series is None:
            return pd.Series(index=df.index, dtype=float, data=float("nan"))

        # 各バーの日付を取得
        bar_dates = pd.Series(
            [idx.date() if hasattr(idx, "date") else idx for idx in df.index],
            index=df.index,
        )

        # 日ごとの最後のATR値を取得
        atr_with_date = pd.DataFrame({"atr": atr_series, "date": bar_dates})
        day_last_atr = atr_with_date.dropna(subset=["atr"]).groupby("date")["atr"].last()

        # 各バーに対応する日のATRをマッピング
        daily_atr = bar_dates.map(day_last_atr)
        daily_atr.index = df.index
        return daily_atr

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        df = df.copy()

        use_orb = p.get("use_orb", False)
        macd_confirm = p.get("macd_confirm", False)
        entry_start_time = p.get("entry_start_time", 935)
        entry_end_time = p.get("entry_end_time", 1300)

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

        # ATR（ボラティリティフィルター用）
        atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr14"] = atr_series if atr_series is not None else float("nan")

        # 出来高移動平均（ブレイクアウト時の急増確認用）
        df["vol_ma20"] = df["volume"].rolling(20).mean()

        min_atr_pct = p.get("min_atr_pct", 0.005)
        vol_ratio = p.get("volume_confirm_ratio", 1.5)

        signals = []
        for i in range(len(df)):
            score = 0.0
            reasons = []

            close = df["close"].iloc[i]

            # --- ATRフィルター（必須条件）---
            atr_val = df["atr14"].iloc[i]
            if pd.isna(atr_val) or close <= 0 or atr_val / close < min_atr_pct:
                signals.append(Signal(0, "LowATR"))
                continue

            # --- 出来高フィルター（必須条件）---
            vol_ma = df["vol_ma20"].iloc[i]
            if pd.isna(vol_ma) or vol_ma <= 0 or df["volume"].iloc[i] < vol_ma * vol_ratio:
                signals.append(Signal(0, "NoVol"))
                continue

            # --- エントリー時間帯フィルター ---
            bar_idx = df.index[i]
            if hasattr(bar_idx, "hour"):
                t = bar_idx.hour * 100 + bar_idx.minute
                if t < entry_start_time or t > entry_end_time:
                    signals.append(Signal(0, "TimeOut"))
                    continue

            # --- ブレイクアウト判定 ---
            breakout_up = not pd.isna(df["high_max"].iloc[i]) and close > df["high_max"].iloc[i]
            breakout_down = not pd.isna(df["low_min"].iloc[i]) and close < df["low_min"].iloc[i]
            orb_up = (
                use_orb
                and "orb_high" in df.columns
                and not pd.isna(df["orb_high"].iloc[i])
                and close > df["orb_high"].iloc[i]
            )
            orb_down = (
                use_orb
                and "orb_low" in df.columns
                and not pd.isna(df["orb_low"].iloc[i])
                and close < df["orb_low"].iloc[i]
            )

            if breakout_up or orb_up:
                score = 1.0
                if breakout_up:
                    reasons.append("ChBreakout")
                if orb_up:
                    reasons.append("ORB_Up")
                # MACD確認ボーナス
                if macd_confirm and "macd_line" in df.columns:
                    ml = df["macd_line"].iloc[i]
                    ms = df["macd_signal"].iloc[i]
                    if not pd.isna(ml) and not pd.isna(ms) and ml > ms:
                        score += 0.5
                        reasons.append("MACD_Bull")
                # 両方成立ボーナス
                if breakout_up and orb_up:
                    score += 0.5
                    reasons.append("DualBreak")
                reasons.extend(["ATR_OK", "VolOK"])
            elif breakout_down or orb_down:
                score = -1.0
                if breakout_down:
                    reasons.append("ChBreakdown")
                if orb_down:
                    reasons.append("ORB_Down")
                # MACD確認ボーナス
                if macd_confirm and "macd_line" in df.columns:
                    ml = df["macd_line"].iloc[i]
                    ms = df["macd_signal"].iloc[i]
                    if not pd.isna(ml) and not pd.isna(ms) and ml < ms:
                        score -= 0.5
                        reasons.append("MACD_Bear")
                # 両方成立ボーナス
                if breakout_down and orb_down:
                    score -= 0.5
                    reasons.append("DualBreak")
                reasons.extend(["ATR_OK", "VolOK"])

            signals.append(Signal(score, " ".join(reasons)))

        return pd.Series(signals, index=df.index)
