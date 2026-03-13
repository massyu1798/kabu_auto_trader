"""
バックテスト v18: MR AND条件化 + 出来高必須 + 閾値厳格化
- MR (ミーンリバージョン): 3条件のうち2条件以上AND必須, 出来高必須, 閾値引き上げ
- BO (ブレイクアウト): v17より停止継続（PF 0.41のため）
- ONG (オーバーナイト・ギャップ): 日足独立ループ, 引け買い→翌朝寄り決済
"""

import pandas as pd
import numpy as np
import yfinance as yf
import yaml
import pandas_ta as ta
from backtest.engine import BacktestEngine, BacktestResult, Trade, Side
from backtest.reporter import generate_report, plot_equity_curve
from backtest.screener import screen_stocks
from strategy.ensemble import EnsembleEngine
from strategy.overnight_gap import OvernightGap
from strategy.breakout import Breakout

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
        # VWAPを計算して追加
        data["vwap"] = ta.vwap(data["high"], data["low"], data["close"], data["volume"])
        # VWAP乖離z-scoreを計算 (MRエグジットのVWAP回帰チェック用)
        vwap_dev = data["close"] - data["vwap"]
        vwap_dev_std = vwap_dev.rolling(window=20, min_periods=5).std()
        data["vwap_z"] = vwap_dev / vwap_dev_std.replace(0, np.nan)
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

def apply_v15_filters(signals_df):
    """
    v16フィルター:
    1. 時間帯フィルター: 9:30〜14:30 以外はHOLD
    2. VWAPフィルター: BO-BUYのみ close < vwap でHOLD（MR-BUYは許可）
    3. SHORT全面禁止: SELL → HOLD
    """
    result = signals_df.copy()
    for idx in result.index:
        # 1. 時間帯フィルター（9:30〜14:30）
        t = idx.hour * 100 + idx.minute
        if not (930 <= t <= 1430):
            result.loc[idx, "final_signal"] = "HOLD"
            continue

        # 2. SHORT全面禁止（v16: 勝率25%のため当面停止）
        if result.loc[idx, "final_signal"] == "SELL":
            result.loc[idx, "final_signal"] = "HOLD"
            continue

        # 3. VWAPフィルター（BO-BUYのみ適用、MR-BUYは除外）
        if result.loc[idx, "final_signal"] == "BUY":
            if "vwap" in result.columns and not pd.isna(result.loc[idx, "vwap"]):
                if result.loc[idx, "close"] < result.loc[idx, "vwap"]:
                    # BOが支配的かMRと同等の場合はHOLD、MRが支配的な場合は逆張り買いを許可
                    mr_score = abs(float(result.loc[idx].get("MeanReversion_score", 0) or 0))
                    bo_score = abs(float(result.loc[idx].get("Breakout_score", 0) or 0))
                    if mr_score <= bo_score:
                        # BOが支配的（またはスコア同等） → VWAP下での買い禁止
                        result.loc[idx, "final_signal"] = "HOLD"
                    # else: MRが支配的 → VWAP下での逆張り買いOK

    return result

def run_ong_backtest(tickers: list, config: dict) -> list:
    """
    オーバーナイト・ギャップ独立バックテスト（日足）
    - IBS < 0.2 + RSI(2) < 10 + 当日下落率 -1.5%以上 → 当日引け買い
    - 翌営業日の寄り付きで成行売り
    - 金曜引け→月曜寄りは不参加
    """
    ong_cfg = config.get("strategies", {}).get("overnight_gap", {})
    ong_params = ong_cfg.get("params", {})
    alloc_pct = config.get("capital_allocation", {}).get("overnight_gap", 0.20)
    initial_capital = config["global"]["initial_capital"]
    ong_capital = alloc_pct * initial_capital        # ONG資金配分
    score_threshold = ong_params.get("signal_score_threshold", 0.8)
    slippage = config["global"].get("slippage_rate", 0.001)
    commission = config["global"].get("commission_rate", 0.0)

    strategy = OvernightGap(ong_params)
    ong_trades = []

    for ticker in tickers:
        df_daily = load_daily(ticker)
        if df_daily is None or len(df_daily) < 10:
            continue

        signals = strategy.generate_signals(df_daily)

        for i in range(len(df_daily) - 1):
            sig = signals.iloc[i]
            # 買いシグナルのみ（score > threshold）
            if sig.score < score_threshold:
                continue

            date = df_daily.index[i]
            # 金曜日は除外（週末リスク回避）
            if hasattr(date, "weekday") and date.weekday() == 4:
                continue

            entry_price = float(df_daily["close"].iloc[i])
            exit_price = float(df_daily["open"].iloc[i + 1])
            exit_date = df_daily.index[i + 1]

            if entry_price <= 0 or exit_price <= 0:
                continue

            # ポジションサイズ: ONG資金の最大30%/銘柄
            max_per_trade = ong_capital * 0.30
            size = (int(max_per_trade / entry_price) // 100) * 100
            if size < 100:
                continue

            ep = entry_price * (1 + slippage)
            xp = exit_price * (1 - slippage)
            pnl = (xp - ep) * size
            pnl -= abs(xp * size * commission)
            pnl_pct = (xp - ep) / ep * 100

            trade = Trade(
                ticker=ticker,
                side=Side.LONG,
                entry_price=ep,
                exit_price=xp,
                entry_date=date,
                exit_date=exit_date,
                size=size,
                pnl=pnl,
                pnl_pct=pnl_pct,
                entry_reason=f"ONG:{sig.reason}",
                exit_reason="翌日寄り決済",
                strategy_tag="overnight_gap",
            )
            ong_trades.append(trade)

    return ong_trades


def main():
    print("=" * 60)
    print("  🌅 日本株自動売買 v18: MR AND条件化 + 出来高必須 + 閾値厳格化")
    print("=" * 60)

    with open("config/strategy_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("\n■ 銘柄スクリーニング...")
    selected = screen_stocks(config)
    tickers = [s["ticker"] for s in selected]

    print(f"\n■ 解析中...")
    ensemble = EnsembleEngine("config/strategy_config.yaml")
    # BO用にORBレンジを計算してsignals_dfに追加するためのインスタンス
    bo_params = config.get("strategies", {}).get("breakout", {}).get("params", {})
    bo_helper = Breakout(bo_params)
    orb_minutes = bo_params.get("orb_minutes", 30)
    signals_dict = {}

    for ticker in tickers:
        df_5m = load_intraday(ticker)
        df_daily = load_daily(ticker)
        if df_5m is None or df_daily is None or len(df_5m) < 20: continue

        bias = calc_daily_bias(df_daily, config)
        signals_df = ensemble.generate_ensemble_signals(df_5m)
        signals_df = apply_v11_filter(signals_df, bias)
        signals_df = apply_v15_filters(signals_df)       # ★ v16フィルター（9:30〜14:30、SHORT禁止、VWAP戦略別）

        # MR用: vwap_z, vwapをsignals_dfに引き継ぎ（ORB計算とは独立）
        if "vwap_z" in df_5m.columns:
            signals_df["vwap_z"] = df_5m["vwap_z"]
        if "vwap" in df_5m.columns:
            signals_df["vwap"] = df_5m["vwap"]

        # BO用: ORBレンジをsignals_dfに追加（エンジンのORB復帰損切りに使用）
        try:
            orb_df = bo_helper._calc_orb(df_5m, minutes=orb_minutes)
            signals_df["orb_high"] = orb_df["orb_high"]
            signals_df["orb_low"] = orb_df["orb_low"]
        except Exception:
            pass

        signals_dict[ticker] = signals_df
        print(".", end="", flush=True)

    print(f"\n\n■ イントラデイ バックテスト実行 (MR+BO)...")
    engine = BacktestEngine("config/strategy_config.yaml")
    result = engine.run(signals_dict)

    print(f"\n■ オーバーナイト・ギャップ バックテスト実行 (ONG)...")
    ong_trades = run_ong_backtest(tickers, config)
    print(f"  ONG取引数: {len(ong_trades)}件")

    # 結果統合
    all_trades = result.trades + ong_trades
    combined_result = BacktestResult(
        trades=all_trades,
        equity_curve=result.equity_curve,
        dates=result.dates,
    )

    print(generate_report(combined_result, engine.initial_capital))

    if all_trades:
        days = len(set(
            t.entry_date.date() if hasattr(t.entry_date, "date") else t.entry_date
            for t in all_trades
        ))
        freq = len(all_trades) / days if days > 0 else 0
        print(f"  🔥 1日平均取引数: {freq:.2f} 回")
        win_trades = [t for t in all_trades if t.pnl > 0]
        print(f"  📊 トータル勝率: {(len(win_trades)/len(all_trades)*100):.1f}%")

    plot_equity_curve(result, engine.initial_capital)
    print("\n✅ v18 テスト完了。")

if __name__ == "__main__":
    main()
