"""戦略③ オーバーナイト・ギャップ戦略 v2.0
- IBS (Internal Bar Strength) + RSI(2) + 当日下落率 の3条件ANDで引け買い
- 翌営業日の寄り付きで決済（引け買い → 翌日寄り売り）
- 金曜・日銀/FOMC当日は不参加
- main_backtest_combined.py から日足ベースで呼び出す想定
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from strategy.base import StrategyBase, Signal

# 日銀金融政策決定会合・FOMC当日除外カレンダー（バックテスト用近似リスト）
BOJ_FOMC_DATES = {
    # 日銀 2024
    "2024-01-23", "2024-03-19", "2024-04-26", "2024-06-14",
    "2024-07-31", "2024-09-20", "2024-10-31", "2024-12-19",
    # FOMC 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-09-18", "2024-11-07", "2024-12-18",
    # 日銀 2025
    "2025-01-24", "2025-03-19", "2025-04-30", "2025-06-17",
    "2025-07-31", "2025-09-19", "2025-10-29", "2025-12-19",
    # FOMC 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    # 日銀 2026
    "2026-01-23", "2026-03-19", "2026-04-28", "2026-06-17",
    # FOMC 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
}


class OvernightGap(StrategyBase):
    def __init__(self, params: dict):
        super().__init__("OvernightGap", params)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        日足データに対してONGシグナルを生成する。

        エントリー条件（全3条件AND）:
          1. IBS = (終値 - 安値) / (高値 - 安値) < ibs_oversold (default 0.2)
          2. RSI(2) < rsi2_oversold (default 10)
          3. 当日下落率 <= min_daily_drop_pct (default -1.5%)

        除外フィルター:
          - 金曜日（週末リスク回避）
          - 日銀金融政策決定会合・FOMC当日

        戻り値: pd.Series[Signal]（スコア=0 → HOLD, スコア >= threshold → BUY候補）
        """
        p = self.params
        df = df.copy()

        ibs_oversold = p.get("ibs_oversold", 0.2)
        rsi2_oversold = p.get("rsi2_oversold", 10)
        min_drop = p.get("min_daily_drop_pct", -1.5)
        score_threshold = p.get("signal_score_threshold", 1.1)
        skip_fomc_boj = p.get("skip_fomc_boj", True)

        # IBS = (Close - Low) / (High - Low)
        hl_range = df["high"] - df["low"]
        df["ibs"] = (df["close"] - df["low"]) / hl_range.replace(0, np.nan)

        # RSI(2) - 超短期RSI
        df["rsi2"] = ta.rsi(df["close"], length=2)

        # 当日リターン
        df["daily_return"] = df["close"].pct_change() * 100

        signals = []
        for i in range(len(df)):
            idx = df.index[i]
            date_str = str(idx.date()) if hasattr(idx, "date") else str(idx)
            ibs = df["ibs"].iloc[i]
            rsi2 = df["rsi2"].iloc[i]
            daily_ret = df["daily_return"].iloc[i]

            if pd.isna(ibs) or pd.isna(rsi2) or pd.isna(daily_ret):
                signals.append(Signal(0.0, ""))
                continue

            # 金曜日フィルター（週末リスク回避）
            if hasattr(idx, "weekday") and idx.weekday() == 4:
                signals.append(Signal(0.0, "Friday:skip"))
                continue

            # 日銀/FOCMカレンダーフィルター
            if skip_fomc_boj and date_str in BOJ_FOMC_DATES:
                signals.append(Signal(0.0, "BOJ/FOMC:skip"))
                continue

            score = 0.0
            reason_parts = []

            # 条件1: IBS < threshold（引け値が安値圏）
            if ibs < ibs_oversold:
                score += 0.5
                reason_parts.append(f"IBS={ibs:.2f}")

            # 条件2: RSI(2) < threshold（極端な売られすぎ）
            if rsi2 < rsi2_oversold:
                score += 0.4
                reason_parts.append(f"RSI2={rsi2:.0f}")

            # 条件3: 当日下落率が min_drop 以上の下落
            if daily_ret <= min_drop:
                score += 0.3
                reason_parts.append(f"Drop={daily_ret:.1f}%")

            signals.append(Signal(score, " ".join(reason_parts)))

        return pd.Series(signals, index=df.index)
