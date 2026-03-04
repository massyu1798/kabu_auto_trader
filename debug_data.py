"""データの中身を確認するスクリプト"""

import yfinance as yf

ticker = "7203.T"
print(f"{ticker} の5分足データを取得中...\n")

data = yf.download(ticker, period="60d", interval="5m")

print(f"取得行数: {len(data)}")
print(f"カラム: {data.columns.tolist()}")
print(f"インデックスの型: {type(data.index)}")

if len(data) > 0:
    print(f"\nタイムゾーン: {data.index.tz}")
    print(f"\n--- 先頭5行 ---")
    print(data.head())
    print(f"\n--- 末尾5行 ---")
    print(data.tail())
    print(f"\n--- 時刻の例（先頭10件） ---")
    for i in range(min(10, len(data))):
        idx = data.index[i]
        print(f"  {idx} | hour={idx.hour} minute={idx.minute}")
else:
    print("\nデータが0行です。")