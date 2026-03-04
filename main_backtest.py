"""
バックテスト v12: モーニング・モメンタム 安定版
"""

import pandas as pd
import yfinance as yf
import yaml
import pandas_ta as ta
from backtest.engine import BacktestEngine
from backtest.reporter import generate_report, plot_equity_curve
from backtest.screener import screen_stocks
from strategy.ensemble import EnsembleEngine

def load_intraday(ticker):
    try:
        data = yf.download(ticker, period="60d", interval="5m", progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0].lower() for col in data.columns]
        else:
            data.columns = [col.lower() for col in data.columns]
        data = data[["open", "high", "low", "close", "volume"]].copy()
        data.dropna(inplace=True)
        if data.index.tz is not None:
            data.index = data.index.tz_convert("Asia/Tokyo")
        else:
            data.index = data.index.tz_localize("UTC").tz_convert("Asia/Tokyo")
        # 取引時間フィルター
        data = data[(data.index.hour >= 9) & ((data.index.hour < 15) | ((data.index.hour == 15) & (data.index.minute <= 25)))]
        return data
    except Exception as e:
        print(f"  ❌ {ticker} 取得エラー: {e}")
        return None

def load_daily(ticker):
    try:
        data = yf.download(ticker, period="120d", interval="1d", progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0].lower() for col in data.columns]
        else:
            data.columns = [col.lower() for col in data.columns]
        return data
    except Exception: return None

def calc_daily_bias(daily_df, config):
    dc = config["daily_bias"]
    df = daily_df.copy()
    df["ema_s"] = ta.ema(df["close"], length=dc["ema_short"])
    df["ema_l"] = ta.ema(df["close"], length=dc["ema_long"])
    bias = {}
    for idx, row in df.iterrows():
        date = idx.date() if hasattr(idx, "date") else idx
        if pd.isna(row["ema_s"]) or pd.isna(row["ema_l"]):
            bias[date] = "NEUTRAL"
        elif row["ema_s"] > row["ema_l"]:
            bias[date] = "BULL"
        else:
            bias[date] = "BEAR"
    return bias

def apply_v11_filter(signals_df, daily_bias):
    """
    v11戦略フィルター: BEAR（下降トレンド）の日は取引禁止
    """
    result = signals_df.copy()
    for idx in result.index:
        date = idx.date() if hasattr(idx, "date") else idx
        b = daily_bias.get(date, "NEUTRAL")
        if b == "BEAR":
            result.loc[idx, "final_signal"] = "HOLD"
    return result

def main():
    print("=" * 60)
    print("  ⚖️ 日本株自動売買 v12 安定版")
    print("=" * 60)

    with open("config/strategy_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("\n■ 銘柄スクリーニング...")
    selected = screen_stocks(config)
    tickers = [s["ticker"] for s in selected]

    print(f"\n■ 解析中...")
    ensemble = EnsembleEngine("config/strategy_config.yaml")
    signals_dict = {}
    
    for ticker in tickers:
        df_5m = load_intraday(ticker)
        df_daily = load_daily(ticker)
        if df_5m is None or df_daily is None or len(df_5m) < 20: continue
        
        bias = calc_daily_bias(df_daily, config)
        signals_df = ensemble.generate_ensemble_signals(df_5m)
        signals_df = apply_v11_filter(signals_df, bias)
        
        signals_dict[ticker] = signals_df
        print(".", end="", flush=True)

    print("\n\n■ バックテスト実行...")
    engine = BacktestEngine("config/strategy_config.yaml")
    result = engine.run(signals_dict)

    print(generate_report(result, engine.initial_capital))
    plot_equity_curve(result, engine.initial_capital)
    print("\n✅ 完了")

if __name__ == "__main__":
    main()