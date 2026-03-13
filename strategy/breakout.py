"""戦略② ORBブレイクアウト・モメンタム v19
- Opening Range Breakout (寄り30分レンジ)
- チャネルブレイクアウト（従来版）との併用
- MACD確認フィルター付き
- ATRフィルター: 低ボラ相場のダマシを排除
- 出来高急増確認（AND必須条件）: ブレイクアウト時に出来高が平均の1.5倍以上
- ORBレンジ幅フィルター: 狭すぎるレンジのダマシを抑制
- ブレイクアウト確認足: 方向に2本連続で引けた場合のみ追加スコア
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
        volume_confirm_ratio = p.get("volume_confirm_ratio", 1.5)
        min_atr_pct = p.get("min_atr_pct", 0.005)
        orb_min_range_ratio = p.get("orb_min_range_ratio", 0.3)
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
        df["vol_ma5"] = df["volume"].rolling(5).mean()

        # 前日ATR（ORBレンジ幅フィルター用）
        df["prev_day_atr"] = self._calc_prev_day_atr(df, atr_period=14)

        signals = []
        for i in range(len(df)):
            score = 0.0
            reasons = []

            close = df["close"].iloc[i]

            # --- エントリー時間帯フィルター ---
            bar_idx = df.index[i]
            if hasattr(bar_idx, "hour"):
                bar_time_int = bar_idx.hour * 100 + bar_idx.minute
                if not (entry_start_time <= bar_time_int <= entry_end_time):
                    signals.append(Signal(0, "TimeFilter"))
                    continue

            # --- ATRフィルター（低ボラ相場のダマシを排除）---
            atr_val = df["atr14"].iloc[i]
            if not pd.isna(atr_val) and close > 0:
                atr_pct = atr_val / close
                if atr_pct < min_atr_pct:
                    signals.append(Signal(0, "LowATR"))
                    continue

            # --- 出来高急増確認（AND必須条件）---
            vol_ma20 = df["vol_ma20"].iloc[i]
            vol_ok = (
                not pd.isna(vol_ma20)
                and vol_ma20 > 0
                and df["volume"].iloc[i] > vol_ma20 * volume_confirm_ratio
            )
            if not vol_ok:
                signals.append(Signal(0, "NoVolSurge"))
                continue

            # --- チャネルブレイクアウト（上方向） ---
            if not pd.isna(df["high_max"].iloc[i]) and close > df["high_max"].iloc[i]:
                score += 0.6
                reasons.append("ChBreakout")

            # --- チャネルブレイクダウン（下方向） ---
            if not pd.isna(df["low_min"].iloc[i]) and close < df["low_min"].iloc[i]:
                score -= 0.6
                reasons.append("ChBreakdown")

            # --- ORBブレイクアウト + レンジ幅フィルター ---
            if use_orb and "orb_high" in df.columns:
                orb_h = df["orb_high"].iloc[i]
                orb_l = df["orb_low"].iloc[i]
                prev_atr = df["prev_day_atr"].iloc[i]

                # ORBレンジ幅フィルター: レンジが前日ATRの orb_min_range_ratio 未満はスコア抑制
                orb_range_ok = True
                if not pd.isna(orb_h) and not pd.isna(orb_l) and not pd.isna(prev_atr) and prev_atr > 0:
                    orb_range = orb_h - orb_l
                    if orb_range < prev_atr * orb_min_range_ratio:
                        orb_range_ok = False

                if orb_range_ok:
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

            # --- ブレイクアウト確認足（2本連続で同方向に引けた場合のみ追加スコア）---
            if i >= 1 and score != 0:
                prev_close = df["close"].iloc[i - 1]
                prev_open = df["open"].iloc[i - 1]
                curr_open = df["open"].iloc[i]
                if score > 0:
                    # 上方向: 前足・現足ともに陽線かつ現足終値が前足終値を上回る
                    if prev_close > prev_open and close > curr_open and close > prev_close:
                        score += 0.2
                        reasons.append("ConfirmBar")
                elif score < 0:
                    # 下方向: 前足・現足ともに陰線かつ現足終値が前足終値を下回る
                    if prev_close < prev_open and close < curr_open and close < prev_close:
                        score -= 0.2
                        reasons.append("ConfirmBar")

            # --- 出来高増加トレンド（補助スコア）---
            if not pd.isna(df["vol_ma5"].iloc[i]) and df["vol_ma5"].iloc[i] > 0:
                if df["volume"].iloc[i] > df["vol_ma5"].iloc[i] * 1.3:
                    if score > 0:
                        score += 0.1
                    elif score < 0:
                        score -= 0.1
                    reasons.append("VolTrend")

            signals.append(Signal(score, " ".join(reasons)))

        return pd.Series(signals, index=df.index)
