"""戦略① VWAP乖離ミーンリバージョン v14
- RSI + ボリンジャーバンド + VWAP乖離の複合シグナル
- 買い(LONG) & 売り(SHORT) 双方向対応
- 出来高急増でセリングクライマックス/バイイングクライマックスを検知
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from strategy.base import StrategyBase, Signal


class MeanReversion(StrategyBase):
    def __init__(self, params: dict):
        super().__init__("MeanReversion", params)

    def _calc_vwap(self, df: pd.DataFrame) -> pd.Series:
        """日中VWAPを算出"""
        vwap = pd.Series(index=df.index, dtype=float)
        for day, group in df.groupby(df.index.date):
            tp = (group["high"] + group["low"] + group["close"]) / 3
            cum_tp_vol = (tp * group["volume"]).cumsum()
            cum_vol = group["volume"].cumsum()
            day_vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
            vwap.loc[group.index] = day_vwap
        return vwap

    def _calc_vwap_deviation(self, df: pd.DataFrame, vwap: pd.Series) -> pd.Series:
        """VWAPからの乖離をσ単位で算出"""
        deviation = df["close"] - vwap
        window = self.params.get("bb_period", 20)
        rolling_std = deviation.rolling(window=window, min_periods=5).std()
        z_score = deviation / rolling_std.replace(0, np.nan)
        return z_score

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        df = df.copy()

        # テクニカル指標算出
        df["rsi"] = ta.rsi(df["close"], length=p.get("rsi_period", 14))

        bb = ta.bbands(df["close"],
                       length=p.get("bb_period", 20),
                       std=p.get("bb_std", 1.5))
        if bb is not None and len(bb.columns) >= 3:
            df["bb_lower"] = bb.iloc[:, 0]
            df["bb_mid"] = bb.iloc[:, 1]
            df["bb_upper"] = bb.iloc[:, 2]
        else:
            df["bb_lower"] = np.nan
            df["bb_mid"] = np.nan
            df["bb_upper"] = np.nan

        # VWAP乖離
        use_vwap = p.get("use_vwap", False)
        if use_vwap:
            df["vwap"] = self._calc_vwap(df)
            df["vwap_z"] = self._calc_vwap_deviation(df, df["vwap"])

        # 出来高フィルター
        vol_ratio = p.get("volume_confirm_ratio", 1.5)
        df["vol_ma"] = df["volume"].rolling(window=20).mean()

        vwap_threshold = p.get("vwap_deviation_threshold", 1.5)

        signals = []
        for i in range(len(df)):
            rsi = df["rsi"].iloc[i]
            close = df["close"].iloc[i]
            bb_lower = df["bb_lower"].iloc[i]
            bb_upper = df["bb_upper"].iloc[i]

            if pd.isna(rsi) or pd.isna(bb_lower):
                signals.append(Signal(0, ""))
                continue

            # 出来高チェック
            vol_ok = (
                not pd.isna(df["vol_ma"].iloc[i])
                and df["vol_ma"].iloc[i] > 0
                and df["volume"].iloc[i] > df["vol_ma"].iloc[i] * vol_ratio
            )

            score = 0.0
            reasons = []

            # === 買いシグナル（売られすぎ → 反発期待） ===
            if rsi < p.get("rsi_oversold", 25):
                score += 0.5
                reasons.append(f"RSI={rsi:.0f}")

            if close <= bb_lower:
                score += 0.5
                reasons.append("BB_Low")

            if use_vwap and not pd.isna(df["vwap_z"].iloc[i]):
                if df["vwap_z"].iloc[i] <= -vwap_threshold:
                    score += 0.5
                    reasons.append(f"VWAP_Z={df['vwap_z'].iloc[i]:.1f}")

            if vol_ok and score > 0:
                score += 0.3
                reasons.append("VolConfirm")

            # === 売りシグナル（買われすぎ → 反落期待） ===
            if rsi > p.get("rsi_overbought", 75):
                score -= 0.5
                reasons.append(f"RSI={rsi:.0f}")

            if close >= bb_upper:
                score -= 0.5
                reasons.append("BB_High")

            if use_vwap and not pd.isna(df["vwap_z"].iloc[i]):
                if df["vwap_z"].iloc[i] >= vwap_threshold:
                    score -= 0.5
                    reasons.append(f"VWAP_Z={df['vwap_z'].iloc[i]:.1f}")

            if vol_ok and score < 0:
                score -= 0.3
                reasons.append("VolConfirm")

            signals.append(Signal(score, " ".join(reasons)))

        return pd.Series(signals, index=df.index)
