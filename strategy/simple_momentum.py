"""
シンプル順張り(値幅 ≥ 3.0%) シグナル生成エンジン

戦略概要:
  - 前場（9:00〜11:25）の騰落率（当日始値 vs 11:25終値）が +3%以上 → ロング
  - 前場の騰落率が -3%以下 → ショート
  - 値幅3%未満の銘柄は対象外（シグナルなし）
  - 候補銘柄を前場出来高の降順でソートし、上位 max_positions_per_side 件を選定
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yaml

# UNIVERSE は strategy/universe.py から import
try:
    from strategy.universe import UNIVERSE
except ImportError:
    from universe import UNIVERSE  # type: ignore

logger = logging.getLogger(__name__)


class SimpleMomentumEngine:
    """シンプル順張り(値幅 ≥ 3.0%)戦略のシグナル生成エンジン。

    使用方法:
        engine = SimpleMomentumEngine("config/simple_momentum_config.yaml")
        result = engine.generate_daily_signal(morning_data_dict, prev_close_dict)
        if result:
            long_tickers, short_tickers = result
    """

    def __init__(self, config_path: str = "config/simple_momentum_config.yaml") -> None:
        """設定ファイルを読み込み、パラメータを初期化する。

        Args:
            config_path: simple_momentum_config.yaml のパス
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        g = self.config["global"]
        self.max_positions_per_side: int = int(g["max_positions_per_side"])

        s = self.config["strategy"]
        # direction: "momentum"=順張り / "meanrev"=逆張り  ★四象限の切替ポイント①
        self.direction: str = str(s.get("direction", "momentum"))
        # min_move_pct: 値幅閾値  ★四象限の切替ポイント②
        self.min_move_pct: float = float(s.get("min_move_pct", 3.0))

    def generate_daily_signal(
        self,
        morning_data: dict[str, pd.DataFrame],
        prev_close: dict[str, float],
    ) -> Optional[tuple[list[str], list[str]]]:
        """前場データからシグナルを生成する。

        各銘柄の前場騰落率（当日始値 vs 11:25終値）を計算し、
        direction と min_move_pct に基づいてロング/ショート候補を選定する。

        Args:
            morning_data: 銘柄→前場5分足DataFrame（JST）の辞書
            prev_close:   銘柄→前日終値の辞書（将来の拡張用に保持）

        Returns:
            (long_tickers, short_tickers) のタプル、またはシグナルなしの場合 None
        """
        candidates: list[dict] = []

        for ticker, df in morning_data.items():
            if df.empty:
                continue

            # 当日始値（9:00バー open）
            day_open = self._get_day_open(df)
            if day_open is None or day_open <= 0:
                continue

            # 11:25バー close（シグナル評価時刻）
            close_1125 = self._get_close_at(df, 11, 25)
            if close_1125 is None:
                # フォールバック: 前場最後のバー
                close_1125 = float(df.iloc[-1]["close"])

            # 前場騰落率（当日始値 vs 11:25終値）
            morning_return_pct = (close_1125 - day_open) / day_open * 100.0

            # 前場出来高（銘柄選定の優先度に使用）
            morning_volume = float(df["volume"].sum())

            candidates.append(
                {
                    "ticker": ticker,
                    "return_pct": morning_return_pct,
                    "volume": morning_volume,
                }
            )

        if not candidates:
            return None

        threshold = self.min_move_pct

        # direction に応じてロング/ショート条件を切り替える ★四象限の切替ポイント①②
        if self.direction == "momentum":
            # 順張り: 上昇銘柄をロング、下落銘柄をショート
            long_candidates = [c for c in candidates if c["return_pct"] >= threshold]
            short_candidates = [c for c in candidates if c["return_pct"] <= -threshold]
        else:
            # 逆張り ("meanrev"): 下落銘柄をロング（反発狙い）、上昇銘柄をショート（過熱冷却）
            long_candidates = [c for c in candidates if c["return_pct"] <= -threshold]
            short_candidates = [c for c in candidates if c["return_pct"] >= threshold]

        # 前場出来高の降順でソート（流動性が高い銘柄を優先）
        long_candidates.sort(key=lambda c: c["volume"], reverse=True)
        short_candidates.sort(key=lambda c: c["volume"], reverse=True)

        # 上位 max_positions_per_side 件を選定
        long_tickers = [c["ticker"] for c in long_candidates[: self.max_positions_per_side]]
        short_tickers = [c["ticker"] for c in short_candidates[: self.max_positions_per_side]]

        if not long_tickers and not short_tickers:
            return None

        return long_tickers, short_tickers

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    @staticmethod
    def _get_day_open(df: pd.DataFrame) -> Optional[float]:
        """前場DataFrameから当日始値（9:00バー open）を取得する。"""
        if df.empty:
            return None
        # 9:00バーを探す
        mask = (df.index.hour == 9) & (df.index.minute == 0)
        bar_900 = df[mask]
        if not bar_900.empty:
            return float(bar_900.iloc[0]["open"])
        # フォールバック: 最初のバーの open
        return float(df.iloc[0]["open"])

    @staticmethod
    def _get_close_at(df: pd.DataFrame, hour: int, minute: int) -> Optional[float]:
        """前場DataFrameから指定時刻のclose価格を取得する。"""
        mask = (df.index.hour == hour) & (df.index.minute == minute)
        bars = df[mask]
        if bars.empty:
            return None
        return float(bars.iloc[0]["close"])
