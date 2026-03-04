"""過去データの取得（Yahoo Finance経由）デイトレ対応版"""

import pandas as pd
import yfinance as yf


def load_stock_data(
    ticker: str,
    start: str = "2025-01-01",
    end: str = "2026-02-19",
    interval: str = "1d",
) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, interval=interval)

    if data.empty:
        raise ValueError(f"データを取得できませんでした: {ticker}")

    data.columns = [
        col[0].lower() if isinstance(col, tuple) else col.lower()
        for col in data.columns
    ]

    data = data[["open", "high", "low", "close", "volume"]].copy()
    data.dropna(inplace=True)

    print(
        f"  📊 {ticker}: {len(data)}行 "
        f"({data.index[0]} ~ {data.index[-1]})"
    )
    return data


def load_daily_data(
    tickers: list[str],
    start: str = "2025-01-01",
    end: str = "2026-02-19",
) -> dict[str, pd.DataFrame]:
    """日足データを一括取得（テクニカル指標の計算用）"""
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = load_stock_data(ticker, start, end, interval="1d")
        except Exception as e:
            print(f"  ⚠️ {ticker}: {e}")
    return data


def load_intraday_data(
    tickers: list[str],
    period: str = "60d",
    interval: str = "5m",
) -> dict[str, pd.DataFrame]:
    """
    分足データを取得（デイトレ用）

    注意: Yahoo Finance の制限
      - 5分足: 直近60日分まで
      - 1分足: 直近7日分まで
      - 15分足: 直近60日分まで
    """
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = load_stock_data(
                ticker, start=None, end=None, interval=interval,
            )
        except Exception as e:
            print(f"  ⚠️ {ticker}: {e}")
    return data


SAMPLE_TICKERS = [
    "7203.T",   # トヨタ自動車
    "6758.T",   # ソニーグループ
    "9984.T",   # ソフトバンクグループ
    "8306.T",   # 三菱UFJ FG
    "6861.T",   # キーエンス
    "9432.T",   # NTT
    "6501.T",   # 日立製作所
    "7974.T",   # 任天堂
    "4063.T",   # 信越化学工業
    "8035.T",   # 東京エレクトロン
    "6902.T",   # デンソー
    "7267.T",   # ホンダ
    "4502.T",   # 武田薬品
    "6098.T",   # リクルート
    "9433.T",   # KDDI
]