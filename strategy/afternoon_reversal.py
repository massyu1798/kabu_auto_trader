"""
午後リバーサル戦略エンジン
- 午前中に動きすぎた銘柄のVWAP回帰を狙う
- エントリー: 12:30〜14:00
- RSI + ボリンジャーバンド + 午前変動率���ィルター
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import yaml


class AfternoonReversalEngine:
    """午後リバーサル専用シグナル生成エンジン"""

    def __init__(self, config_path: str = "config/afternoon_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        a = self.config["afternoon_reversal"]
        self.entry_start = a.get("entry_start", 1230)
        self.entry_end = a.get("entry_end", 1400)
        self.min_morning_move_pct = a.get("min_morning_move_pct", 1.5)
        self.rsi_period = a.get("rsi_period", 14)
        self.rsi_oversold = a.get("rsi_oversold", 25)
        self.rsi_overbought = a.get("rsi_overbought", 75)
        self.bb_period = a.get("bb_period", 20)
        self.bb_std = a.get("bb_std", 2.0)
        self.volume_surge_ratio = a.get("volume_surge_ratio", 1.3)

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

    def _calc_morning_move(self, df: pd.DataFrame) -> pd.Series:
        """午前中の変動率を算出（寄り値からの変動率%）"""
        morning_move = pd.Series(index=df.index, dtype=float, data=0.0)

        for day, group in df.groupby(df.index.date):
            # 当日の始値（寄り値）
            open_price = group["open"].iloc[0]
            if open_price == 0:
                continue
            # 各時点での寄りからの変動率
            move_pct = ((group["close"] - open_price) / open_price) * 100
            morning_move.loc[group.index] = move_pct

        return morning_move

    def _is_entry_time(self, timestamp) -> bool:
        if not hasattr(timestamp, "hour"):
            return False
        t = timestamp.hour * 100 + timestamp.minute
        return self.entry_start <= t <= self.entry_end

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """午���リバーサルシグナル��生成"""
        result = df.copy()

        # インジケーター算出
        result["rsi"] = ta.rsi(result["close"], length=self.rsi_period)

        bb = ta.bbands(result["close"], length=self.bb_period, std=self.bb_std)
        if bb is not None:
            result["bb_upper"] = bb.iloc[:, 2]   # Upper Band
            result["bb_lower"] = bb.iloc[:, 0]   # Lower Band
            result["bb_mid"] = bb.iloc[:, 1]     # Middle Band
        else:
            result["bb_upper"] = np.nan
            result["bb_lower"] = np.nan
            result["bb_mid"] = np.nan

        # VWAP算出
        result["vwap"] = self._calc_vwap(result)

        # 午前変動率
        result["morning_move"] = self._calc_morning_move(result)

        # 出来高の移動平均
        result["vol_ma"] = result["volume"].rolling(window=20).mean()

        # シグナル生成
        result["afternoon_signal"] = "HOLD"

        for idx in result.index:
            if not self._is_entry_time(idx):
                continue

            row = result.loc[idx]

            # データ不足チェック
            if pd.isna(row.get("rsi")) or pd.isna(row.get("bb_lower")) or pd.isna(row.get("vwap")):
                continue

            morning_move = row["morning_move"]
            rsi = row["rsi"]
            close = row["close"]
            vwap = row["vwap"]
            bb_lower = row["bb_lower"]
            bb_upper = row["bb_upper"]
            vol = row["volume"]
            vol_ma = row.get("vol_ma", 0)

            # 出来高フィルター（出来高がある程度ないと信頼性が低い）
            if pd.isna(vol_ma) or vol_ma == 0:
                continue

            # === 買いシグナル（売られすぎからの反発） ===
            if (
                morning_move <= -self.min_morning_move_pct     # 午前中に大きく下落
                and rsi <= self.rsi_oversold                    # RSI売られすぎ
                and close <= bb_lower                           # BB下限以下
                and close < vwap                                # VWAPより下（割安）
            ):
                result.loc[idx, "afternoon_signal"] = "BUY"

            # === 売りシグナル（買われすぎからの反落） ===
            elif (
                morning_move >= self.min_morning_move_pct      # 午前中に大きく上昇
                and rsi >= self.rsi_overbought                  # RSI買われすぎ
                and close >= bb_upper                           # BB上限以上
                and close > vwap                                # VWAPより上（割��）
            ):
                result.loc[idx, "afternoon_signal"] = "SELL"

        return result
