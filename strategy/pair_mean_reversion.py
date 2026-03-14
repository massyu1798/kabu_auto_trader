"""
前場終了時モメンタムスコアによる後場寄りモメンタム継続型ペア戦略
シグナル生成エンジン

戦略概要:
  - 11:25時点の前場5分足データから5種の特徴量を算出
  - 各特徴量をz-score標準化し重み付き合成スコアを計算
  - スコア上位（前場上昇モメンタム強）銘柄をロング、下位（前場下落継続）銘柄をショート
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
    """前場終了時モメンタムスコアによるペア選定エンジン（モメンタム継続戦略）

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
        self.max_topix_late_move_pct: float = float(filt.get("max_topix_late_move_pct", 0.5))
        # 出来高急増フィルタ（新旧パラメータ名の優先順位: max_volume_ratio_exclude > volume_ratio_exclude_threshold > max_volume_ratio）
        exclude_threshold = (
            filt.get("max_volume_ratio_exclude")
            or filt.get("volume_ratio_exclude_threshold")
            or filt.get("max_volume_ratio", 5.0)
        )
        self.volume_ratio_exclude_threshold: float = float(exclude_threshold)
        self.max_volume_ratio: float = self.volume_ratio_exclude_threshold
        # ギャップフィルタ
        self.max_gap_pct: float = float(filt.get("max_gap_pct", 3.0))
        # セクター制御
        self.max_same_sector_total: int = int(filt.get("max_same_sector_total", 2))
        # ★ NEW: 前場レンジ / ATR(14) 比率フィルタ
        self.max_range_atr_ratio: float = float(filt.get("max_range_atr_ratio", 2.0))
        # ★ NEW: 前場終盤（11:15〜11:25）急変フィルタ
        self.max_late_surge_pct: float = float(filt.get("max_late_surge_pct", 1.5))
        # ★ NEW: ロング・ショート間でも同セクター重複を回避
        self.no_cross_sector_overlap: bool = bool(filt.get("no_cross_sector_overlap", False))

        r = self.config.get("risk", {})
        # ★ NEW: ネットエクスポージャ（ロング金額 - ショート金額）±5%以内
        self.max_net_exposure_pct: float = float(r.get("max_net_exposure_pct", 0.05))
        # ★ NEW: β調整オプション
        self.beta_adjust: bool = bool(r.get("beta_adjust", False))

        et = self.config.get("entry_thresholds", {})
        self.long_daily_return_min: float = float(et.get("long_daily_return_min", 0.01))
        self.long_relative_return_min: float = float(et.get("long_relative_return_min", 0.007))
        self.long_late_momentum_min: float = float(et.get("long_late_momentum_min", 0.003))
        self.short_daily_return_max: float = float(et.get("short_daily_return_max", -0.01))
        self.short_relative_return_max: float = float(et.get("short_relative_return_max", -0.007))
        self.short_late_momentum_max: float = float(et.get("short_late_momentum_max", -0.003))
        self.min_volume_ratio: float = float(et.get("min_volume_ratio", 0.5))

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
        atr_dict: Optional[dict[str, float]] = None,
    ) -> pd.DataFrame:
        """全銘柄の特徴量を計算して DataFrame を返す。

        計算する特徴量:
            - daily_return          : 当日騰落率（前日終値比）
            - relative_return       : TOPIX 比相対リターン
            - late_morning_momentum : 前場後半モメンタム（11:00〜11:25）
            - vwap_deviation        : 前場 VWAP 乖離率
            - volume_ratio          : 前場出来高倍率（20 日平均比）
            - morning_turnover      : 前場売買代金（フィルタ用）
            - late_surge_pct        : 前場終盤急変率（11:15〜11:25 / 100）★NEW
            - morning_range         : 前場高値-安値（ATRフィルタ用）★NEW
            - prev_atr              : 前日ATR(14)（ATRフィルタ用）★NEW

        Args:
            morning_data_dict: 銘柄→前場 5 分足 DataFrame の辞書
            prev_close_dict:   銘柄→前日終値の辞書
            topix_return:      TOPIX 前場リターン（小数）
            avg_volume_dict:   銘柄→20 日平均出来高の辞書（省略可）
            atr_dict:          銘柄→前日 ATR(14) の辞書（省略可）★NEW

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

            # ★ NEW: 11:15 バーの close（前場終盤急変フィルタ用）
            close_1115 = self._get_price_at(df, 11, 15)
            if close_1115 is None or close_1115 <= 0:
                close_1115 = self._get_price_at(df, 11, 10)
            if close_1115 is None or close_1115 <= 0:
                close_1115 = close_1100  # フォールバック

            # 当日始値（寄付き価格）：ギャップフィルタ用
            open_price = float(df["open"].iloc[0])

            # 寄付きギャップ（当日始値 - 前日終値）/ 前日終値
            gap_pct = (open_price - prev_close) / prev_close if prev_close > 0 else 0.0

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

            # ★ NEW: 前場終盤急変率（11:15〜11:25）
            late_surge_pct = (
                (close_1125 - close_1115) / close_1115 if close_1115 > 0 else 0.0
            )

            # ★ NEW: 前場レンジ（高値 - 安値）
            morning_high = float(df["high"].max())
            morning_low = float(df["low"].min())
            morning_range = morning_high - morning_low

            # ★ NEW: 前日 ATR(14) — atr_dict から取得（なければ 0）
            prev_atr = (atr_dict or {}).get(ticker, 0.0)

            if volume_ratio >= self.volume_ratio_exclude_threshold:
                logger.debug(
                    f"{ticker}: 出来高倍率 {volume_ratio:.1f}x — 急増銘柄（フィルタ対象）"
                )

            records.append(
                {
                    "ticker": ticker,
                    "name": info["name"],
                    "sector": info["sector"],
                    "close_1125": close_1125,
                    "prev_close": prev_close,
                    "open_price": open_price,
                    "gap_pct": gap_pct,
                    "daily_return": daily_return,
                    "relative_return": relative_return,
                    "late_morning_momentum": late_morning_momentum,
                    "vwap_deviation": vwap_deviation,
                    "volume_ratio": volume_ratio,
                    "morning_turnover": morning_turnover,
                    "morning_volume": morning_volume,
                    "late_surge_pct": late_surge_pct,
                    "morning_range": morning_range,
                    "prev_atr": prev_atr,
                }
            )

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records).set_index("ticker")

    def calc_scores(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """z-score 標準化 → 重み付き合成スコアを計算する。

        スコアが高いほど上昇モメンタム強（ロング候補）、低いほど下落継続（ショート候補）。

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
        features_df: pd.DataFrame,
        topix_return: float,
        topix_late_move: float = 0.0,
    ) -> pd.DataFrame:
        """フィルタを適用し、異常銘柄・低流動性銘柄を除外する。

        適用フィルタ:
            1. TOPIX 地合いフィルタ（前場 ±2% 超で全見送り）
            2. TOPIX 終盤急変フィルタ（11:00〜11:25 の変化 ±0.5% 超で全見送り）
            3. 流動性フィルタ（前場売買代金 ≥ 20 億円）
            4. 値動き異常フィルタ（|当日騰落率| > 5%）
            5. 後半モメンタム異常フィルタ（|後半 momentum| > 3%）
            6. 出来高急増フィルタ（出来高倍率 > volume_ratio_exclude_threshold）
            7. ギャップフィルタ（|寄付きギャップ| > max_gap_pct%）
            8. ★ NEW: ATR 比率フィルタ（前場レンジ >= max_range_atr_ratio × ATR(14)）
            9. ★ NEW: 前場終盤急変フィルタ（|11:15〜11:25| > max_late_surge_pct%）

        Note:
            スコア閾値フィルタは select_candidates → calc_scores 後に適用する。

        Args:
            features_df:      calc_features() の出力 DataFrame
            topix_return:     TOPIX 前場リターン（小数）
            topix_late_move:  TOPIX 11:00〜11:25 の変化率（小数）

        Returns:
            フィルタ後 DataFrame（空の場合はその日全見送り）
        """
        if features_df.empty:
            return features_df.copy()

        # 1. TOPIX 地合いフィルタ
        if abs(topix_return) * 100.0 > self.max_topix_move_pct:
            logger.info(
                f"TOPIX 地合いフィルタ発動: {topix_return * 100:.2f}%"
                f" > ±{self.max_topix_move_pct}% → 本日全見送り"
            )
            return pd.DataFrame()

        # 2. TOPIX 終盤急変フィルタ
        if abs(topix_late_move) * 100.0 > self.max_topix_late_move_pct:
            logger.info(
                f"TOPIX 終盤急変フィルタ発動: {topix_late_move * 100:.2f}%"
                f" > ±{self.max_topix_late_move_pct}% → 本日全見送り"
            )
            return pd.DataFrame()

        result = features_df.copy()

        # 3. 流動性フィルタ
        n_before = len(result)
        result = result[result["morning_turnover"] >= self.min_morning_turnover]
        logger.debug(f"流動性フィルタ: {n_before} → {len(result)} 銘柄")

        # 4. 値動き異常フィルタ
        n_before = len(result)
        result = result[result["daily_return"].abs() * 100.0 <= self.max_daily_return_pct]
        logger.debug(f"騰落率フィルタ: {n_before} → {len(result)} 銘柄")

        # 5. 後半モメンタム異常フィルタ
        n_before = len(result)
        result = result[
            result["late_morning_momentum"].abs() * 100.0 <= self.max_late_momentum_pct
        ]
        logger.debug(f"モメンタムフィルタ: {n_before} → {len(result)} 銘柄")

        # 6. 出来高急増フィルタ（材料株除外: 決算・IR等の代理指標）
        n_before = len(result)
        result = result[result["volume_ratio"] <= self.volume_ratio_exclude_threshold]
        logger.debug(
            f"出来高急増フィルタ: {n_before} → {len(result)} 銘柄 "
            f"({n_before - len(result)}銘柄除外)"
        )

        # 7. ギャップフィルタ（寄付きからのギャップが大きすぎる銘柄を除外）
        if "gap_pct" in result.columns:
            n_before = len(result)
            result = result[result["gap_pct"].abs() * 100.0 <= self.max_gap_pct]
            logger.debug(
                f"ギャップフィルタ: {n_before} → {len(result)} 銘柄 "
                f"({n_before - len(result)}銘柄除外)"
            )

        # 8. ★ NEW: ATR 比率フィルタ（前場高値-安値レンジが前日ATR(14)の2倍以上を除外）
        if "morning_range" in result.columns and "prev_atr" in result.columns:
            n_before = len(result)
            # prev_atr が 0 の場合はフィルタ対象外（データなし）
            has_atr = result["prev_atr"] > 0
            range_ratio = result["morning_range"] / result["prev_atr"].replace(0, float("nan"))
            too_volatile = has_atr & (range_ratio >= self.max_range_atr_ratio)
            result = result[~too_volatile]
            n_removed = n_before - len(result)
            if n_removed > 0:
                logger.debug(
                    f"ATR比率フィルタ: {n_before} → {len(result)} 銘柄 "
                    f"(レンジ/{self.max_range_atr_ratio}×ATR超: {n_removed}銘柄除外)"
                )

        # 9. ★ NEW: 前場終盤急変フィルタ（11:15〜11:25）
        if "late_surge_pct" in result.columns:
            n_before = len(result)
            result = result[
                result["late_surge_pct"].abs() * 100.0 <= self.max_late_surge_pct
            ]
            n_removed = n_before - len(result)
            if n_removed > 0:
                logger.debug(
                    f"前場終盤急変フィルタ: {n_before} → {len(result)} 銘柄 "
                    f"(11:15〜11:25 急変±{self.max_late_surge_pct}%超: {n_removed}銘柄除外)"
                )

        return result

    def select_candidates(
        self,
        filtered_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """entry_thresholds による明示的エントリー条件フィルタで候補を選定する。

        ロング条件（全て満たす銘柄）＝上昇モメンタム追随:
            - daily_return      > long_daily_return_min  (前日比 +1.0% 以上)
            - relative_return   > long_relative_return_min (TOPIX 比 +0.7% 以上)
            - late_morning_momentum > long_late_momentum_min (後半 +0.3% 以上)
            - volume_ratio      >= min_volume_ratio (20 日平均の 0.5 倍以上)

        ショート条件（全て満たす銘柄）＝下落継続売り:
            - daily_return      < short_daily_return_max  (前日比 -1.0% 以下)
            - relative_return   < short_relative_return_max (TOPIX 比 -0.7% 以下)
            - late_morning_momentum < short_late_momentum_max (後半 -0.3% 以下)
            - volume_ratio      >= min_volume_ratio (20 日平均の 0.5 倍以上)

        Args:
            filtered_df: apply_filters() の出力 DataFrame

        Returns:
            (long_pool, short_pool) のタプル（どちらも空の場合あり）
        """
        if filtered_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        long_mask = (
            (filtered_df["daily_return"] > self.long_daily_return_min)
            & (filtered_df["relative_return"] > self.long_relative_return_min)
            & (filtered_df["late_morning_momentum"] > self.long_late_momentum_min)
            & (filtered_df["volume_ratio"] >= self.min_volume_ratio)
        )
        short_mask = (
            (filtered_df["daily_return"] < self.short_daily_return_max)
            & (filtered_df["relative_return"] < self.short_relative_return_max)
            & (filtered_df["late_morning_momentum"] < self.short_late_momentum_max)
            & (filtered_df["volume_ratio"] >= self.min_volume_ratio)
        )

        long_pool = filtered_df[long_mask]
        short_pool = filtered_df[short_mask]

        # long_mask と short_mask は排他的（daily_return > +1% と < -1% は同時不成立）
        # 万が一両側に入った銘柄はいずれかのプールにのみ存在する
        logger.debug(
            f"候補選定: ロング={len(long_pool)}銘柄  ショート={len(short_pool)}銘柄"
        )
        return long_pool, short_pool

    def select_pairs(
        self,
        long_pool: pd.DataFrame,
        short_pool: pd.DataFrame,
        capital: float = 0.0,
        beta_dict: Optional[dict[str, float]] = None,
    ) -> tuple[list[str], list[str]]:
        """セクター偏り制御付きでロング・ショートの銘柄を最終選定する。

        選定ロジック（モメンタム継続）:
            - ロング候補を score 降順（最も上昇モメンタムが強い）で並べ、最大 max_positions_per_side 銘柄
            - ショート候補を score 昇順（最も下落モメンタムが強い）で並べ、最大 max_positions_per_side 銘柄
            - 各サイドで同一セクターは max_same_sector_per_side 件まで
            - no_cross_sector_overlap=True の場合: ロング選定済みセクターはショートに入れない
            - no_cross_sector_overlap=False の場合: ロング+ショート合計で max_same_sector_total 件まで
            - ペアである必要はない。ロングのみ / ショートのみも可。

        Args:
            long_pool:   select_candidates() + calc_scores() 後のロング候補 DataFrame
            short_pool:  select_candidates() + calc_scores() 後のショート候補 DataFrame
            capital:     現在の資本（ネットエクスポージャ計算用、0で無効）
            beta_dict:   銘柄→β値の辞書（beta_adjust=True 時のサイジング用）

        Returns:
            (long_tickers, short_tickers) のタプル
        """
        # まずロングを選定し、そのセクター集計をショート選定に引き継ぐ
        long_sector_count: dict[str, int] = {}
        long_tickers: list[str] = []
        if not long_pool.empty:
            sorted_long = long_pool.sort_values("score", ascending=False)
            for ticker, row in sorted_long.iterrows():
                if len(long_tickers) >= self.max_positions_per_side:
                    break
                sector = str(row.get("sector", "unknown"))
                if long_sector_count.get(sector, 0) >= self.max_same_sector_per_side:
                    continue
                long_tickers.append(str(ticker))
                long_sector_count[sector] = long_sector_count.get(sector, 0) + 1

        # ★ ショート選定: no_cross_sector_overlap に応じてセクター制御を切替
        short_tickers: list[str] = []
        if not short_pool.empty:
            sorted_short = short_pool.sort_values("score", ascending=True)
            short_sector_count: dict[str, int] = {}
            # no_cross_sector_overlap=True → ロング選定済みセクターを完全除外
            # no_cross_sector_overlap=False → 合計カウンタで max_same_sector_total まで許可
            combined_count: dict[str, int] = dict(long_sector_count)

            for ticker, row in sorted_short.iterrows():
                if len(short_tickers) >= self.max_positions_per_side:
                    break
                sector = str(row.get("sector", "unknown"))

                # 片側制約チェック
                if short_sector_count.get(sector, 0) >= self.max_same_sector_per_side:
                    continue

                if self.no_cross_sector_overlap:
                    # ロング・ショート間での同セクター重複を完全回避
                    if sector in long_sector_count:
                        logger.debug(
                            f"{ticker}({sector}): ロング側と同セクター "
                            f"(no_cross_sector_overlap=True) → スキップ"
                        )
                        continue
                else:
                    # 合計制約チェック（ロング+ショート合算）
                    if combined_count.get(sector, 0) >= self.max_same_sector_total:
                        logger.debug(
                            f"{ticker}({sector}): セクター合計上限 "
                            f"{self.max_same_sector_total} に到達 → スキップ"
                        )
                        continue

                short_tickers.append(str(ticker))
                short_sector_count[sector] = short_sector_count.get(sector, 0) + 1
                combined_count[sector] = combined_count.get(sector, 0) + 1

        return long_tickers, short_tickers

    def generate_daily_signal(
        self,
        morning_data_dict: dict[str, pd.DataFrame],
        prev_close_dict: dict[str, float],
        topix_data: Optional[pd.DataFrame],
        avg_volume_dict: Optional[dict[str, float]] = None,
        atr_dict: Optional[dict[str, float]] = None,
        beta_dict: Optional[dict[str, float]] = None,
        capital: float = 0.0,
    ) -> Optional[tuple[list[str], list[str]]]:
        """1 日分のシグナルを生成するラッパーメソッド。

        処理フロー:
            1. TOPIX 前場リターン・終盤変化率を計算
            2. calc_features → apply_filters → select_candidates →
               calc_scores → スコア閾値フィルタ → select_pairs

        Args:
            morning_data_dict: 銘柄→前場 5 分足 DataFrame の辞書
            prev_close_dict:   銘柄→前日終値の辞書（"1306.T" を含む）
            topix_data:        TOPIX ETF（1306.T）の前場 5 分足 DataFrame
            avg_volume_dict:   銘柄→20 日平均出来高の辞書（省略可）
            atr_dict:          銘柄→前日 ATR(14) の辞書（省略可）★NEW
            beta_dict:         銘柄→β値の辞書（beta_adjust=True 時用）★NEW
            capital:           現在の資本（ネットエクスポージャ計算用、0で無効）★NEW

        Returns:
            (long_tickers, short_tickers) または None（シグナルなし）
        """
        # TOPIX 前場リターンと終盤変化率を計算
        topix_return = 0.0
        topix_late_move = 0.0
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
                topix_close_1125 = close_p if topix_close_1125 is None else topix_close_1125

            # TOPIX 終盤変化（11:00〜11:25）
            topix_close_1100 = self._get_price_at(topix_data, 11, 0)
            if topix_close_1100 is None or topix_close_1100 <= 0:
                topix_close_1100 = self._get_price_at(topix_data, 10, 55)
            if (
                topix_close_1125 is not None
                and topix_close_1100 is not None
                and topix_close_1100 > 0
            ):
                topix_late_move = (topix_close_1125 - topix_close_1100) / topix_close_1100

        logger.debug(
            f"TOPIX 前場リターン: {topix_return * 100:.2f}%"
            f"  終盤変化: {topix_late_move * 100:.2f}%"
        )

        # 特徴量計算（★ atr_dict を追加）
        features_df = self.calc_features(
            morning_data_dict, prev_close_dict, topix_return, avg_volume_dict, atr_dict
        )
        if features_df.empty:
            logger.debug("特徴量が空: 対象銘柄なし")
            return None

        # フィルタ適用（地合い・流動性・値動き異常・出来高急増・ATR・終盤急変）
        filtered_df = self.apply_filters(features_df, topix_return, topix_late_move)
        if filtered_df.empty:
            logger.debug("全銘柄フィルタアウト")
            return None

        # 候補選定（entry_thresholds による明示的フィルタ）
        long_pool, short_pool = self.select_candidates(filtered_df)
        if long_pool.empty and short_pool.empty:
            logger.debug("候補銘柄なし（エントリー閾値）")
            return None

        # スコア計算（候補内でのランキング用）
        combined = pd.concat([long_pool, short_pool]).drop_duplicates()
        scored = self.calc_scores(combined)

        # スコア閾値フィルタ
        scored = scored[scored["score"].abs() >= self.min_abs_score_threshold]
        if scored.empty:
            logger.debug("スコア閾値フィルタで全銘柄除外")
            return None

        # スコア後のロング/ショートプールを再構築
        long_scored = scored[scored.index.isin(long_pool.index)]
        short_scored = scored[scored.index.isin(short_pool.index)]

        # 最終選定（セクター偏り制御付き）
        long_tickers, short_tickers = self.select_pairs(
            long_scored, short_scored, capital=capital, beta_dict=beta_dict
        )

        if not long_tickers and not short_tickers:
            logger.debug("選定銘柄なし（スコア・セクター制約）")
            return None

        logger.debug(
            f"選定 → LONG={long_tickers}, SHORT={short_tickers}"
            f" | ロング候補={len(long_pool)}, ショート候補={len(short_pool)}"
        )
        return long_tickers, short_tickers
