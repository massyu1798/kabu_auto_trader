"""シグナルが出ない原因を調査するスクリプト（修正版）"""

import pandas as pd
import yfinance as yf
from strategy.ensemble import EnsembleEngine


def load_one_stock(ticker):
    data = yf.download(ticker, period="60d", interval="5m")
    if data.empty:
        raise ValueError(f"データなし: {ticker}")

    # MultiIndexカラム解消
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0].lower() for col in data.columns]
    else:
        data.columns = [col.lower() for col in data.columns]

    data = data[["open", "high", "low", "close", "volume"]].copy()
    data.dropna(inplace=True)

    # UTCから日本時間に変換
    if data.index.tz is not None:
        data.index = data.index.tz_convert("Asia/Tokyo")
    else:
        data.index = data.index.tz_localize("UTC").tz_convert("Asia/Tokyo")

    # 東証の取引時間のみ
    data = data[
        (data.index.hour >= 9) & (
            (data.index.hour < 15) |
            ((data.index.hour == 15) & (data.index.minute <= 25))
        )
    ]
    return data


def main():
    ticker = "7203.T"
    print(f"{ticker} の5分足データを取得中...\n")
    df = load_one_stock(ticker)
    print(f"データ行数: {len(df)}")
    print(f"期間: {df.index[0]} ~ {df.index[-1]}\n")

    ensemble = EnsembleEngine("config/strategy_config.yaml")
    result = ensemble.generate_ensemble_signals(df)

    print("=" * 50)
    print("各戦略のスコア分布")
    print("=" * 50)

    for col in result.columns:
        if col.endswith("_score"):
            scores = result[col]
            non_zero = scores[scores != 0]
            print(f"\n{col}:")
            print(f"  全行数:        {len(scores)}")
            print(f"  スコア!=0の行: {len(non_zero)}")
            if len(non_zero) > 0:
                print(f"  最小値:        {non_zero.min():.2f}")
                print(f"  最大値:        {non_zero.max():.2f}")
                for val in sorted(non_zero.unique()):
                    count = (non_zero == val).sum()
                    print(f"    {val:+.1f}: {count}回")

    print(f"\n{'=' * 50}")
    print("統合スコア (ensemble_score)")
    print("=" * 50)
    es = result["ensemble_score"]
    non_zero_es = es[es != 0]
    print(f"  全行数:        {len(es)}")
    print(f"  スコア!=0の行: {len(non_zero_es)}")
    if len(non_zero_es) > 0:
        print(f"  最小値:        {non_zero_es.min():.2f}")
        print(f"  最大値:        {non_zero_es.max():.2f}")
        for val in sorted(non_zero_es.unique()):
            count = (non_zero_es == val).sum()
            print(f"    {val:+.1f}: {count}回")

    print(f"\n{'=' * 50}")
    print("最終シグナル")
    print("=" * 50)
    print(result["final_signal"].value_counts().to_string())

    print(f"\n{'=' * 50}")
    print("統合スコア 上位/下位5件")
    print("=" * 50)
    top = result.nlargest(5, "ensemble_score")
    bottom = result.nsmallest(5, "ensemble_score")

    print("\n【買い方向 上位5件】")
    for idx, row in top.iterrows():
        print(f"  {idx} | score={row['ensemble_score']:+.1f} | signal={row['final_signal']}")

    print("\n【売り方向 上位5件】")
    for idx, row in bottom.iterrows():
        print(f"  {idx} | score={row['ensemble_score']:+.1f} | signal={row['final_signal']}")


if __name__ == "__main__":
    main()