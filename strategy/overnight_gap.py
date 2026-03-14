"""
オーバーナイト・ギャップ (ONG) シグナル生成エンジン v1.0

エントリー条件（引け買い → 翌日寄り売り）:
  1. IBS = (終値 - 安値) / (高値 - 安値) < 0.2  (当日引け値が安値圏)
  2. RSI(2) < 10                                 (極端な売られ過ぎ)
  3. 日経225 ETF(1321.T) 翌日始値 > 当日終値 × (1 + threshold%) (夜間追い風近似)
  4. 当日下落率 <= decline_threshold%            (例: -1.5%)
  5. 金曜フィルター: 週末リスク回避
  6. 日銀金融政策決定会合・FOMC 当日は除外
"""

from datetime import date

import pandas as pd
import pandas_ta as ta

# ────────────────────────────────────────────────────────────────
# ブラックアウト日: 日銀金融政策決定会合・FOMC 当日
# ────────────────────────────────────────────────────────────────
BLACKOUT_DATES: frozenset = frozenset([
    # BOJ 2024
    date(2024, 1, 23), date(2024, 3, 19), date(2024, 4, 26), date(2024, 6, 14),
    date(2024, 7, 31), date(2024, 9, 20), date(2024, 10, 31), date(2024, 12, 19),
    # BOJ 2025
    date(2025, 1, 24), date(2025, 3, 19), date(2025, 4, 30), date(2025, 5, 1),
    date(2025, 6, 17), date(2025, 7, 31), date(2025, 9, 19),
    # FOMC 2024
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1), date(2024, 6, 12),
    date(2024, 7, 31), date(2024, 9, 18), date(2024, 11, 7), date(2024, 12, 18),
    # FOMC 2025
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17),
])


def _build_night_tailwind_map(
    nikkei_etf_df: pd.DataFrame | None,
    threshold_pct: float,
) -> dict:
    """
    1321.T 日足データから「夜間追い風マップ」を生成。
    date -> bool: 翌日始値が当日終値対比 +threshold_pct% 以上ならTrue。
    """
    if nikkei_etf_df is None or nikkei_etf_df.empty:
        return {}

    etf = nikkei_etf_df[["open", "close"]].dropna().copy()
    night_tailwind: dict = {}
    for i in range(len(etf) - 1):
        today_idx = etf.index[i]
        today_date = today_idx.date() if hasattr(today_idx, "date") else today_idx
        today_close = etf.iloc[i]["close"]
        next_open = etf.iloc[i + 1]["open"]
        gap_pct = (next_open / today_close - 1) * 100
        night_tailwind[today_date] = gap_pct >= threshold_pct
    return night_tailwind


def generate_ong_signals(
    ticker_daily: dict,
    nikkei_etf_df: pd.DataFrame | None,
    config: dict,
) -> dict:
    """
    各銘柄の日足 DataFrame に ONG_signal 列を付与して返す。

    Parameters
    ----------
    ticker_daily : dict[str, pd.DataFrame]
        銘柄コード -> 日足 OHLCV (columns: open, high, low, close, volume)
    nikkei_etf_df : pd.DataFrame | None
        1321.T の日足データ（夜間追い風近似用）
    config : dict
        overnight_config.yaml の内容

    Returns
    -------
    dict[str, pd.DataFrame]
        ONG_signal 列を追加した DataFrame の辞書
    """
    params = config.get("ong", {})
    ibs_threshold = float(params.get("ibs_threshold", 0.2))
    rsi2_threshold = float(params.get("rsi2_threshold", 10.0))
    decline_threshold = float(params.get("decline_threshold", -1.5))
    night_tailwind_threshold = float(params.get("night_tailwind_threshold", 0.3))
    skip_friday = bool(params.get("skip_friday", True))

    night_tailwind = _build_night_tailwind_map(nikkei_etf_df, night_tailwind_threshold)
    has_tailwind_data = len(night_tailwind) > 0

    result: dict = {}

    for ticker, df in ticker_daily.items():
        if df is None or df.empty or len(df) < 5:
            continue

        d = df.copy()

        # ── IBS ──────────────────────────────────────────────────
        hl_range = d["high"] - d["low"]
        d["ibs"] = (d["close"] - d["low"]) / hl_range.replace(0.0, float("nan"))

        # ── RSI(2) ───────────────────────────────────────────────
        d["rsi2"] = ta.rsi(d["close"], length=2)

        # ── 当日下落率 ──────────────────────────────────────────
        d["day_return_pct"] = d["close"].pct_change() * 100

        # ── シグナル生成 ─────────────────────────────────────────
        d["ONG_signal"] = False

        for i in range(1, len(d)):
            row = d.iloc[i]
            idx = d.index[i]
            trade_date = idx.date() if hasattr(idx, "date") else idx

            # 1. IBS 条件
            if pd.isna(row["ibs"]) or row["ibs"] >= ibs_threshold:
                continue

            # 2. RSI(2) 条件
            if pd.isna(row["rsi2"]) or row["rsi2"] >= rsi2_threshold:
                continue

            # 3. 下落率条件
            if pd.isna(row["day_return_pct"]) or row["day_return_pct"] > decline_threshold:
                continue

            # 4. 夜間追い風条件 (1321.T データがある場合のみ適用)
            if has_tailwind_data:
                if not night_tailwind.get(trade_date, False):
                    continue

            # 5. 金曜フィルター
            if skip_friday and hasattr(idx, "weekday") and idx.weekday() == 4:
                continue

            # 6. ブラックアウト日フィルター
            if trade_date in BLACKOUT_DATES:
                continue

            d.at[idx, "ONG_signal"] = True

        result[ticker] = d

    return result
