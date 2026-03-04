"""銘柄自動スクリーニング: 流動性・ボラティリティで動的に選定（エラー修正版）"""

import pandas as pd
import yfinance as yf
import pandas_ta as ta

STOCK_POOL = [
    "7203.T", "6758.T", "9984.T", "8306.T", "6861.T",
    "9432.T", "6501.T", "7974.T", "4063.T", "8035.T",
    "6902.T", "7267.T", "4502.T", "6098.T", "9433.T",
    "6762.T", "6857.T", "4568.T", "6367.T", "6954.T",
    "8058.T", "8316.T", "9983.T", "6503.T", "7741.T",
    "4661.T", "6981.T", "3382.T", "8801.T", "5401.T",
    "2914.T", "6273.T", "4519.T", "7751.T", "6702.T",
    "8766.T", "9022.T", "4307.T", "6752.T", "3659.T",
]

def screen_stocks(config: dict, period: str = "60d") -> list[dict]:
    sc = config.get("screening", {})
    min_volume = sc.get("min_volume", 300000)
    min_price = sc.get("min_price", 300)
    max_price = sc.get("max_price", 20000)
    min_atr_pct = sc.get("min_atr_pct", 0.7)
    max_stocks = sc.get("max_stocks", 50)

    candidates = []
    for ticker in STOCK_POOL:
        try:
            data = yf.download(ticker, period="30d", interval="1d", progress=False)
            if data is None or data.empty or len(data) < 20: continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0].lower() for col in data.columns]
            else:
                data.columns = [col.lower() for col in data.columns]

            last_close = float(data["close"].iloc[-1])
            avg_volume = float(data["volume"].iloc[-10:].mean())

            # ATR算出のエラーチェック強化
            atr_series = ta.atr(data["high"], data["low"], data["close"], length=14)
            if atr_series is None or len(atr_series) == 0 or pd.isna(atr_series.iloc[-1]):
                continue
            
            last_atr = float(atr_series.iloc[-1])
            atr_pct = (last_atr / last_close) * 100

            if avg_volume < min_volume or last_close < min_price or last_close > max_price or atr_pct < min_atr_pct:
                continue

            candidates.append({
                "ticker": ticker,
                "close": last_close,
                "volume": avg_volume,
                "atr": last_atr,
                "atr_pct": atr_pct,
                "score": atr_pct * (avg_volume / 1000000),
            })
        except Exception: continue

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:max_stocks]