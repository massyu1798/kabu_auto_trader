"""戦略① VWAP乖離ミーンリバージョン v18
- RSI + ボリンジャーバンド + VWAP乖離の複合シグナル（AND条件化）
- 買い(LONG) & 売り(SHORT) 双方向対応
- 3条件のうち2条件以上の同時成立でシグナル発行（質の低いシグナルを排除）
- 出来高確認をエントリー必須条件に変更（板の薄い場面のダマシを回避）
- BBスクイーズ時はシグナルを抑制
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

    def _rsi_buy_score(self, rsi: float, oversold: float) -> float:
        """RSI段階的スコア（買い）: oversold付近でグラデーション"""
        if rsi < oversold - 10:
            return 1.0
        elif rsi < oversold:
            return 0.5
        return 0.0

    def _rsi_sell_score(self, rsi: float, overbought: float) -> float:
        """RSI段階的スコア（売り）: overbought付近でグラデーション"""
        if rsi > overbought + 10:
            return 1.0
        elif rsi > overbought:
            return 0.5
        return 0.0

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        df = df.copy()

        # テクニカル指標算出
        df["rsi"] = ta.rsi(df["close"], length=p.get("rsi_period", 14))

        bb_std = p.get("bb_std", 2.0)
        bb = ta.bbands(df["close"],
                       length=p.get("bb_period", 20),
                       std=bb_std)
        if bb is not None and len(bb.columns) >= 3:
            df["bb_lower"] = bb.iloc[:, 0]
            df["bb_mid"] = bb.iloc[:, 1]
            df["bb_upper"] = bb.iloc[:, 2]
        else:
            df["bb_lower"] = np.nan
            df["bb_mid"] = np.nan
            df["bb_upper"] = np.nan

        # BBバンド幅（スクイーズ検出）
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
        bb_width_min = df["bb_width"].rolling(window=20, min_periods=5).min()

        # VWAP乖離
        use_vwap = p.get("use_vwap", False)
        if use_vwap:
            df["vwap"] = self._calc_vwap(df)
            df["vwap_z"] = self._calc_vwap_deviation(df, df["vwap"])

        # 出来高フィルター
        vol_ratio = p.get("volume_confirm_ratio", 1.3)
        df["vol_ma"] = df["volume"].rolling(window=20).mean()

        vwap_threshold = p.get("vwap_deviation_threshold", 1.2)
        rsi_oversold = p.get("rsi_oversold", 30)
        rsi_overbought = p.get("rsi_overbought", 70)
        has_vwap_z = use_vwap and "vwap_z" in df.columns

        signals = []
        for i in range(len(df)):
            rsi = df["rsi"].iloc[i]
            close = df["close"].iloc[i]
            bb_lower = df["bb_lower"].iloc[i]
            bb_upper = df["bb_upper"].iloc[i]

            if pd.isna(rsi) or pd.isna(bb_lower):
                signals.append(Signal(0, ""))
                continue

            # BBスクイーズ判定: バンド幅が直近最小値付近ならシグナル抑制
            bw = df["bb_width"].iloc[i]
            bw_min = bb_width_min.iloc[i]
            if not pd.isna(bw) and not pd.isna(bw_min) and bw_min > 0:
                if bw <= bw_min * 1.1:
                    signals.append(Signal(0, "BB_Squeeze"))
                    continue

            # 出来高チェック（エントリー必須条件）
            vol_ok = (
                not pd.isna(df["vol_ma"].iloc[i])
                and df["vol_ma"].iloc[i] > 0
                and df["volume"].iloc[i] > df["vol_ma"].iloc[i] * vol_ratio
            )

            # 出来高条件を満たさない場合はシグナルなし（板の薄い場面のダマシを回避）
            if not vol_ok:
                signals.append(Signal(0, "NoVol"))
                continue

            score = 0.0
            reasons = []

            # === 買いシグナル（売られすぎ → 反発期待）: 3条件のうち2条件以上で発行 ===
            buy_conditions = 0
            buy_reasons = []

            rsi_buy = self._rsi_buy_score(rsi, rsi_oversold)
            if rsi_buy > 0:
                buy_conditions += 1
                buy_reasons.append(f"RSI={rsi:.0f}")

            if close <= bb_lower:
                buy_conditions += 1
                buy_reasons.append("BB_Low")

            vwap_z = df["vwap_z"].iloc[i] if has_vwap_z else float("nan")
            if has_vwap_z and not pd.isna(vwap_z) and vwap_z <= -vwap_threshold:
                buy_conditions += 1
                buy_reasons.append(f"VWAP_Z={vwap_z:.1f}")

            if buy_conditions >= 2:
                score = 0.5 * buy_conditions  # 2条件=1.0, 3条件=1.5
                if buy_conditions == 3:
                    score += 0.3  # 3条件ボーナス（高確度シグナル）
                reasons = buy_reasons + ["VolOK"]
            else:
                # === 売りシグナル（買われすぎ → 反落期待）: 3条件のうち2条件以上で発行 ===
                sell_conditions = 0
                sell_reasons = []

                rsi_sell = self._rsi_sell_score(rsi, rsi_overbought)
                if rsi_sell > 0:
                    sell_conditions += 1
                    sell_reasons.append(f"RSI={rsi:.0f}")

                if close >= bb_upper:
                    sell_conditions += 1
                    sell_reasons.append("BB_High")

                if has_vwap_z and not pd.isna(vwap_z) and vwap_z >= vwap_threshold:
                    sell_conditions += 1
                    sell_reasons.append(f"VWAP_Z={vwap_z:.1f}")

                if sell_conditions >= 2:
                    score = -(0.5 * sell_conditions)  # 2条件=-1.0, 3条件=-1.5
                    if sell_conditions == 3:
                        score -= 0.3  # 3条件ボーナス（高確度シグナル）
                    reasons = sell_reasons + ["VolOK"]

            signals.append(Signal(score, " ".join(reasons)))

        return pd.Series(signals, index=df.index)
