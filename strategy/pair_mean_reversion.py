"""
前場終了時相対過熱・過冷えスコアによる後場寄り平均回帰型ペア戦略
シグナル生成エンジン

戦略概要:
  - 11:25時点の前場5分足データから5種の特徴量を算出
  - 各特徴量をz-score標準化し重み付き合成スコアを計算
  - スコア上位（過熱）銘柄をショート、下位（過冷え）銘柄をロング
  - セクター偏り制御・流動性・値動き異常フィルタを適用
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ユニバース定義（36銘柄、Yahoo Finance ticker形式）
UNIVERSE: dict[str, dict[str, str]] = {
    "9984.T": {"name": "ソフトバンクG",        "sector": "通信"},
    "8306.T": {"name": "三菱UFJ FG",           "sector": "銀行"},
    "5401.T": {"name": "日本製鉄",              "sector": "鉄鋼"},
    "7203.T": {"name": "トヨタ自動車",          "sector": "自動車"},
    "6758.T": {"name": "ソニーG",               "sector": "電機"},
    "6501.T": {"name": "日立製作所",            "sector": "電機"},
    "8316.T": {"name": "三井住友FG",            "sector": "銀行"},
    "6762.T": {"name": "TDK",                   "sector": "電子部品"},
    "6857.T": {"name": "アドバンテスト",        "sector": "半導体"},
    "6981.T": {"name": "村田製作所",            "sector": "電子部品"},
    "7267.T": {"name": "ホンダ",                "sector": "自動車"},
    "6702.T": {"name": "富士通",                "sector": "電機"},
    "8058.T": {"name": "三菱商事",              "sector": "商社"},
    "4063.T": {"name": "信越化学",              "sector": "化学"},
    "6752.T": {"name": "パナソニックHD",        "sector": "電機"},
    "6098.T": {"name": "リクルートHD",          "sector": "サービス"},
    "7974.T": {"name": "任天堂",                "sector": "ゲーム"},
    "6503.T": {"name": "三菱電機",              "sector": "電機"},
    "4307.T": {"name": "野村総研",              "sector": "IT"},
    "8801.T": {"name": "三井不動産",            "sector": "不動産"},
    "6954.T": {"name": "ファナック",            "sector": "FA"},
    "6902.T": {"name": "デンソー",              "sector": "自動車部品"},
    "4568.T": {"name": "第一三共",              "sector": "医薬"},
    "3382.T": {"name": "セブン＆アイ",          "sector": "小売"},
    "9433.T": {"name": "KDDI",                  "sector": "通信"},
    "8766.T": {"name": "東京海上HD",            "sector": "保険"},
    "8035.T": {"name": "東京エレクトロン",      "sector": "半導体"},
    "4661.T": {"name": "オリエンタルランド",    "sector": "サービス"},
    "3659.T": {"name": "ネクソン",              "sector": "ゲーム"},
    "4502.T": {"name": "武田薬品",              "sector": "医薬"},
    "2914.T": {"name": "JT",                    "sector": "食品"},
    "4519.T": {"name": "中外製薬",              "sector": "医薬"},
    "7751.T": {"name": "キヤノン",              "sector": "電機"},
    "9022.T": {"name": "JR東海",               "sector": "運輸"},
    "7741.T": {"name": "HOYA",                  "sector": "精密"},
    "6367.T": {"name": "ダイキン工業",          "sector": "機械"},
}


class PairMeanReversionEngine:
    """前場終了時相対過熱・過冷えスコアによるペア選定エンジン

    使用方法:
        engine = PairMeanReversionEngine("config/pair_meanrev_config.yaml")
        result = engine.generate_daily_signal(
            morning_data_dict, prev_close_dict, topix_morning_df, avg_volume_dict
        )
        if result:
            long_tickers, short_tickers = result
    """

    def __init__(self, config_path: str = "config/pair_meanrev_config.yaml") -> None:
        """設定ファイルを読み込み、パラメータを初期化する。

        Args:
            config_path: pair_meanrev_config.yaml のパス
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        g = self.config["global"]
        self.initial_capital: float = float(g["initial_capital"])
        self.max_positions_per_side: int = int(g["max_positions_per_side"])

        sc = self.config["scoring"]
        self.weights: dict[str, float] = dict(sc["weights"])
        self.min_abs_score_threshold: float = float(sc["min_abs_score_threshold"])

        filt = self.config["filters"]
        self.min_morning_turnover: float = float(filt["min_morning_turnover"])
        self.max_daily_return_pct: float = float(filt["max_daily_return_pct"])
        self.max_late_momentum_pct: float = float(filt["max_late_momentum_pct"])
        self.max_topix_move_pct: float = float(filt["max_topix_move_pct"])
        self.max_same_sector_per_side: int = int(filt["max_same_sector_per_side"])
        self.max_volume_ratio_alert: float = float(filt.get("max_volume_ratio_alert", 3.0))

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _get_price_at(
        self, df: pd.DataFrame, hour: int, minute: int
    ) -> Optional[float]:
        """指定時刻（hour:minute）のclose価格を取得する。

        指定時刻のバーが存在しない場合は、その直前の最も近いバーを返す。

        Args:
            df: タイムゾーン付き DatetimeIndex を持つ 5 分足 DataFrame
            hour: 時
            minute: 分

        Returns:
            close 価格、または取得できない場合は None
        """
        if df is None or df.empty:
            return None
        try:
            exact_mask = (df.index.hour == hour) & (df.index.minute == minute)
            exact = df[exact_mask]
            if not exact.empty:
                return float(exact["close"].iloc[-1])

            # 直前バーにフォールバック
            target_mins = hour * 60 + minute
            bar_mins = df.index.hour * 60 + df.index.minute
            before = df[bar_mins <= target_mins]
            if not before.empty:
                return float(before["close"].iloc[-1])
        except Exception as exc:
            logger.debug(f"_get_price_at({hour}:{minute:02d}) 失敗: {exc}")
        return None

    def _calc_morning_vwap(self, df: pd.DataFrame) -> float:
        """前場 VWAP（出来高加重平均価格）を算出する。

        Args:
            df: 前場 5 分足 DataFrame（9:00〜11:25）

        Returns:
            VWAP 値（出来高がゼロの場合は最終 close）
        """
        if df.empty:
            return 0.0
        cum_vol = float(df["volume"].sum())
        if cum_vol == 0.0:
            return float(df["close"].iloc[-1])
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        return float((typical * df["volume"]).sum() / cum_vol)

    # ------------------------------------------------------------------
    # 公開メソッド
    # ------------------------------------------------------------------

    def calc_features(
        self,
        morning_data_dict: dict[str, pd.DataFrame],
        prev_close_dict: dict[str, float],
        topix_return: float,
        avg_volume_dict: Optional[dict[str, float]] = None,
    ) -> pd.DataFrame:
        """全銘柄の特徴量を計算して DataFrame を返す。

        計算する特徴量:
            - daily_return          : 当日騰落率（前日終値比）
            - relative_return       : TOPIX 比相対リターン
            - late_morning_momentum : 前場後半モメンタム（11:00〜11:25）
            - vwap_deviation        : 前場 VWAP 乖離率
            - volume_ratio          : 前場出来高倍率（20 日平均比）
            - morning_turnover      : 前場売買代金（フィルタ用）

        Args:
            morning_data_dict: 銘柄→前場 5 分足 DataFrame の辞書
            prev_close_dict:   銘柄→前日終値の辞書
            topix_return:      TOPIX 前場リターン（小数）
            avg_volume_dict:   銘柄→20 日平均出来高の辞書（省略可）

        Returns:
            特徴量 DataFrame（index = ticker）
        """
        records = []

        for ticker, info in UNIVERSE.items():
            df = morning_data_dict.get(ticker)
            if df is None or df.empty:
                continue

            prev_close = prev_close_dict.get(ticker)
            if prev_close is None or prev_close <= 0:
                continue

            # 11:25 バーの close（前場引け価格の代理）
            close_1125 = self._get_price_at(df, 11, 25)
            if close_1125 is None or close_1125 <= 0:
                logger.debug(f"{ticker}: 11:25 バーが取得できません")
                continue

            # 11:00 バーの close（前場後半モメンタム計算用）
            close_1100 = self._get_price_at(df, 11, 0)
            if close_1100 is None or close_1100 <= 0:
                close_1100 = self._get_price_at(df, 10, 55)
            if close_1100 is None or close_1100 <= 0:
                # フォールバック: 当日の最初のバー close
                close_1100 = float(df["close"].iloc[0])

            # 1. 当日騰落率
            daily_return = (close_1125 - prev_close) / prev_close

            # 2. TOPIX 比相対リターン
            relative_return = daily_return - topix_return

            # 3. 前場後半モメンタム（11:00〜11:25）
            late_morning_momentum = (
                (close_1125 - close_1100) / close_1100 if close_1100 > 0 else 0.0
            )

            # 4. 前場 VWAP 乖離率
            morning_vwap = self._calc_morning_vwap(df)
            vwap_deviation = (
                (close_1125 - morning_vwap) / morning_vwap if morning_vwap > 0 else 0.0
            )

            # 5. 出来高倍率（前場出来高 / 20 日平均出来高）
            morning_volume = float(df["volume"].sum())
            avg_vol = (avg_volume_dict or {}).get(ticker)
            volume_ratio = (morning_volume / avg_vol) if (avg_vol and avg_vol > 0) else 1.0

            # 前場売買代金（フィルタ用）
            morning_turnover = float((df["close"] * df["volume"]).sum())

            if volume_ratio >= self.max_volume_ratio_alert:
                logger.info(
                    f"{ticker}: 出来高倍率 {volume_ratio:.1f}x — 要注意"
                )

            records.append(
                {
                    "ticker": ticker,
                    "name": info["name"],
                    "sector": info["sector"],
                    "close_1125": close_1125,
                    "prev_close": prev_close,
                    "daily_return": daily_return,
                    "relative_return": relative_return,
                    "late_morning_momentum": late_morning_momentum,
                    "vwap_deviation": vwap_deviation,
                    "volume_ratio": volume_ratio,
                    "morning_turnover": morning_turnover,
                    "morning_volume": morning_volume,
                }
            )

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records).set_index("ticker")

    def calc_scores(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """z-score 標準化 → 重み付き合成スコアを計算する。

        スコアが高いほど過熱（ショート候補）、低いほど過冷え（ロング候補）。

        Args:
            features_df: calc_features() の出力 DataFrame

        Returns:
            z-score 列・score 列を追加した DataFrame
        """
        if features_df.empty:
            return features_df.copy()

        result = features_df.copy()

        feature_cols = [
            "daily_return",
            "relative_return",
            "late_morning_momentum",
            "vwap_deviation",
            "volume_ratio",
        ]

        # 各特徴量を z-score 化
        for col in feature_cols:
            z_col = f"z_{col}"
            if col not in result.columns:
                result[z_col] = 0.0
                continue
            vals = result[col].dropna()
            if len(vals) < 2:
                result[z_col] = 0.0
                continue
            mu = float(vals.mean())
            sigma = float(vals.std())
            if sigma == 0.0:
                result[z_col] = 0.0
            else:
                result[z_col] = (result[col] - mu) / sigma

        # 重み付き合成スコア（不足列は 0 埋め）
        w = self.weights
        score = pd.Series(0.0, index=result.index)
        for col_name, weight in [
            ("daily_return", w.get("daily_return", 0.30)),
            ("relative_return", w.get("relative_return", 0.25)),
            ("late_morning_momentum", w.get("late_morning_momentum", 0.20)),
            ("vwap_deviation", w.get("vwap_deviation", 0.15)),
            ("volume_ratio", w.get("volume_ratio", 0.10)),
        ]:
            z_col = f"z_{col_name}"
            if z_col in result.columns:
                score = score + weight * result[z_col].fillna(0.0)

        result["score"] = score
        return result

    def apply_filters(
        self,
        scores_df: pd.DataFrame,
        features_df: pd.DataFrame,
        topix_return: float,
    ) -> pd.DataFrame:
        """フィルタを適用し、異常銘柄・低流動性銘柄を除外する。

        適用フィルタ:
            1. TOPIX 地合いフィルタ（前場 ±2% 超で全見送り）
            2. 流動性フィルタ（前場売買代金 ≥ 20 億円）
            3. 値動き異常フィルタ（|当日騰落率| > 5%）
            4. 後半モメンタム異常フィルタ（|後半 momentum| > 3%）
            5. スコア閾値フィルタ（|score| < min_abs_score_threshold）

        Args:
            scores_df:    calc_scores() の出力 DataFrame
            features_df:  calc_features() の出力 DataFrame（参照用）
            topix_return: TOPIX 前場リターン（小数）

        Returns:
            フィルタ後 DataFrame（空の場合はその日全見送り）
        """
        if scores_df.empty:
            return scores_df.copy()

        # 1. TOPIX 地合いフィルタ
        if abs(topix_return) * 100.0 > self.max_topix_move_pct:
            logger.info(
                f"TOPIX 地合いフィルタ発動: {topix_return * 100:.2f}%"
                f" > ±{self.max_topix_move_pct}% → 本日全見送り"
            )
            return pd.DataFrame()

        result = scores_df.copy()

        # 2. 流動性フィルタ
        n_before = len(result)
        result = result[result["morning_turnover"] >= self.min_morning_turnover]
        logger.debug(f"流動性フィルタ: {n_before} → {len(result)} 銘柄")

        # 3. 値動き異常フィルタ
        n_before = len(result)
        result = result[result["daily_return"].abs() * 100.0 <= self.max_daily_return_pct]
        logger.debug(f"騰落率フィルタ: {n_before} → {len(result)} 銘柄")

        # 4. 後半モメンタム異常フィルタ
        n_before = len(result)
        result = result[
            result["late_morning_momentum"].abs() * 100.0 <= self.max_late_momentum_pct
        ]
        logger.debug(f"モメンタムフィルタ: {n_before} → {len(result)} 銘柄")

        # 5. スコア閾値フィルタ
        n_before = len(result)
        result = result[result["score"].abs() >= self.min_abs_score_threshold]
        logger.debug(f"スコア閾値フィルタ: {n_before} → {len(result)} 銘柄")

        return result

    def select_pairs(
        self,
        filtered_df: pd.DataFrame,
    ) -> tuple[list[str], list[str]]:
        """セクター偏り制御付きでロング・ショートの銘柄を選定する。

        選定ロジック:
            - score 上位（正値）→ ショート候補（過熱）
            - score 下位（負値）→ ロング候補（過冷え）
            - 各サイドで同一セクターは max_same_sector_per_side 件まで

        Args:
            filtered_df: apply_filters() の出力 DataFrame

        Returns:
            (long_tickers, short_tickers) のタプル
        """
        if filtered_df.empty:
            return [], []

        sorted_desc = filtered_df.sort_values("score", ascending=False)

        # ショート候補: score > 0（過熱）
        short_pool = sorted_desc[sorted_desc["score"] > 0]
        # ロング候補: score < 0（過冷え）、score の小さい順
        long_pool = sorted_desc[sorted_desc["score"] < 0].sort_values(
            "score", ascending=True
        )

        def _select(pool: pd.DataFrame, n: int) -> list[str]:
            selected: list[str] = []
            sector_count: dict[str, int] = {}
            for ticker, row in pool.iterrows():
                if len(selected) >= n:
                    break
                sector = str(row.get("sector", "unknown"))
                if sector_count.get(sector, 0) < self.max_same_sector_per_side:
                    selected.append(str(ticker))
                    sector_count[sector] = sector_count.get(sector, 0) + 1
            return selected

        short_tickers = _select(short_pool, self.max_positions_per_side)
        long_tickers = _select(long_pool, self.max_positions_per_side)

        return long_tickers, short_tickers

    def generate_daily_signal(
        self,
        morning_data_dict: dict[str, pd.DataFrame],
        prev_close_dict: dict[str, float],
        topix_data: Optional[pd.DataFrame],
        avg_volume_dict: Optional[dict[str, float]] = None,
    ) -> Optional[tuple[list[str], list[str]]]:
        """1 日分のシグナルを生成するラッパーメソッド。

        処理フロー:
            1. TOPIX 前場リターンを計算
            2. calc_features → calc_scores → apply_filters → select_pairs

        Args:
            morning_data_dict: 銘柄→前場 5 分足 DataFrame の辞書
            prev_close_dict:   銘柄→前日終値の辞書（"1306.T" を含む）
            topix_data:        TOPIX ETF（1306.T）の前場 5 分足 DataFrame
            avg_volume_dict:   銘柄→20 日平均出来高の辞書（省略可）

        Returns:
            (long_tickers, short_tickers) または None（シグナルなし）
        """
        # TOPIX 前場リターンを計算
        topix_return = 0.0
        if topix_data is not None and not topix_data.empty:
            topix_close_1125 = self._get_price_at(topix_data, 11, 25)
            topix_prev = prev_close_dict.get("1306.T")
            if topix_close_1125 and topix_prev and topix_prev > 0:
                topix_return = (topix_close_1125 - topix_prev) / topix_prev
            else:
                # フォールバック: 当日始値からの変動率
                open_p = float(topix_data["open"].iloc[0])
                close_p = float(topix_data["close"].iloc[-1])
                if open_p > 0:
                    topix_return = (close_p - open_p) / open_p

        logger.debug(f"TOPIX 前場リターン: {topix_return * 100:.2f}%")

        # 特徴量計算
        features_df = self.calc_features(
            morning_data_dict, prev_close_dict, topix_return, avg_volume_dict
        )
        if features_df.empty:
            logger.debug("特徴量が空: 対象銘柄なし")
            return None

        # スコア計算
        scores_df = self.calc_scores(features_df)

        # フィルタ適用
        filtered_df = self.apply_filters(scores_df, features_df, topix_return)
        if filtered_df.empty:
            logger.debug("全銘柄フィルタアウト")
            return None

        # ペア選定
        long_tickers, short_tickers = self.select_pairs(filtered_df)

        if not long_tickers and not short_tickers:
            logger.debug("選定銘柄なし（スコア・セクター制約）")
            return None

        logger.debug(
            f"選定 → LONG={long_tickers}, SHORT={short_tickers}"
            f" | 通過銘柄数={len(filtered_df)}"
        )
        return long_tickers, short_tickers
