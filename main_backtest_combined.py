"""
合体バックテスト: 午前 v12.4 + 午後リバーサル v1.2
- 午前（9:00-12:00）: モーニング・モメンタム順張り
- 午後（12:30-14:00）: アフタヌーン・リバーサル逆張り
"""

import pandas as pd
import yfinance as yf
import yaml
import pandas_ta as ta
from backtest.engine import BacktestEngine
from backtest.afternoon_engine import AfternoonBacktestEngine
from backtest.screener import screen_stocks
from strategy.ensemble import EnsembleEngine
from strategy.afternoon_reversal import AfternoonReversalEngine


def load_intraday(ticker):
    try:
        data = yf.download(ticker, period="60d", interval="5m", progress=False)
        if data.empty:
            return None
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
        data = data[
            (data.index.hour >= 9)
            & ((data.index.hour < 15) | ((data.index.hour == 15) & (data.index.minute <= 25)))
        ]
        return data
    except Exception as e:
        print(f"  x {ticker}: {e}")
        return None


def load_daily(ticker):
    try:
        data = yf.download(ticker, period="120d", interval="1d", progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0].lower() for col in data.columns]
        else:
            data.columns = [col.lower() for col in data.columns]
        return data
    except Exception:
        return None


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
    result = signals_df.copy()
    for idx in result.index:
        date = idx.date() if hasattr(idx, "date") else idx
        b = daily_bias.get(date, "NEUTRAL")
        if b == "BEAR":
            result.loc[idx, "final_signal"] = "HOLD"
    return result


def format_report_section(title, trades, initial_capital, equity_curve):
    if not trades:
        return f"\n  {title}: トレードなし\n"

    total_trades = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / total_trades * 100

    total_pnl = sum(t.pnl for t in trades)
    total_return = (total_pnl / initial_capital) * 100

    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

    total_win_amt = sum(t.pnl for t in wins)
    total_loss_amt = abs(sum(t.pnl for t in losses))
    pf = (total_win_amt / total_loss_amt) if total_loss_amt != 0 else float("inf")

    max_dd = 0
    if equity_curve:
        peak = equity_curve[0]
        for e in equity_curve:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd

    return f"""
  [{title}]
    純損益:         {total_pnl:>+14,.0f} 円 ({total_return:+.2f}%)
    最大DD:         {max_dd:>13.2f} %
    トレード数:     {total_trades} (勝ち{len(wins)} / 負け{len(losses)})
    勝率:           {win_rate:.1f}%
    平均利益:       {avg_win:>+14,.0f} 円
    平均損失:       {avg_loss:>+14,.0f} 円
    PF:             {pf:.2f}
"""


def main():
    print("=" * 60)
    print("  午前v12.4 + 午後リバーサルv1.2 合体バックテスト")
    print("=" * 60)

    # === 設定読み込み ===
    with open("config/strategy_config.yaml", "r", encoding="utf-8") as f:
        morning_config = yaml.safe_load(f)
    with open("config/afternoon_config.yaml", "r", encoding="utf-8") as f:
        afternoon_config = yaml.safe_load(f)

    initial_capital = morning_config["global"]["initial_capital"]

    # === 銘柄スクリーニング ===
    print("\n■ 銘柄スクリーニング...")
    selected = screen_stocks(morning_config)
    tickers = [s["ticker"] for s in selected]
    print(f"  -> {len(tickers)}銘柄を選定")

    # === データ取得 + シグナル生成 ===
    print("\n■ 解析中...")

    # 午前用
    ensemble = EnsembleEngine("config/strategy_config.yaml")
    morning_signals = {}

    # 午後用
    reversal_engine = AfternoonReversalEngine("config/afternoon_config.yaml")
    afternoon_signals = {}

    for ticker in tickers:
        df_5m = load_intraday(ticker)
        df_daily = load_daily(ticker)
        if df_5m is None or len(df_5m) < 20:
            continue

        # 午前シグナル
        if df_daily is not None:
            bias = calc_daily_bias(df_daily, morning_config)
            signals_df = ensemble.generate_ensemble_signals(df_5m)
            signals_df = apply_v11_filter(signals_df, bias)
            morning_signals[ticker] = signals_df

        # 午後シグナル
        afternoon_df = reversal_engine.generate_signals(df_5m)
        afternoon_signals[ticker] = afternoon_df

        print(".", end="", flush=True)

    print(f"\n  -> 午前: {len(morning_signals)}銘柄 / 午後: {len(afternoon_signals)}銘柄")

    # === バックテスト実行 ===
    print("\n■ バックテスト実行...")

    # 午前
    print("  [午前 v12.4] 実行中...")
    morning_engine = BacktestEngine("config/strategy_config.yaml")
    morning_result = morning_engine.run(morning_signals)

    # 午後
    print("  [午後 リバーサル v1.2] 実行中...")
    afternoon_engine = AfternoonBacktestEngine("config/afternoon_config.yaml")
    afternoon_result = afternoon_engine.run(afternoon_signals)

    # === 合算 ===
    all_trades = morning_result.trades + afternoon_result.trades
    total_pnl = sum(t.pnl for t in all_trades)
    total_return = (total_pnl / initial_capital) * 100
    equity_final = initial_capital + total_pnl

    morning_pnl = sum(t.pnl for t in morning_result.trades)
    afternoon_pnl = sum(t.pnl for t in afternoon_result.trades)

    # 合算DD（簡易: 各セッ��ョンの最大DDの大きい方）
    def calc_dd(equity_curve):
        if not equity_curve:
            return 0
        peak = equity_curve[0]
        max_dd = 0
        for e in equity_curve:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    morning_dd = calc_dd(morning_result.equity_curve)
    afternoon_dd = calc_dd(afternoon_result.equity_curve)

    # === レポート出力 ===
    report = f"""
============================================================
  午前v12.4 + 午後リバーサルv1.2 合体レポート
============================================================

■ 総合成績
  初期資金:       {initial_capital:>14,.0f} 円
  最終資産:       {equity_final:>14,.0f} 円
  純損益:         {total_pnl:>+14,.0f} 円 ({total_return:+.2f}%)
  総トレード数:   {len(all_trades)}
    午前:         {len(morning_result.trades)}��� -> {morning_pnl:>+,.0f} 円
    午後:         {len(afternoon_result.trades)}件 -> {afternoon_pnl:>+,.0f} 円

■ セッション別詳��
{format_report_section("午前 v12.4 モーニング・モメンタム", morning_result.trades, initial_capital, morning_result.equity_curve)}
{format_report_section("午後 v1.2 アフ���ヌーン・リバーサル", afternoon_result.trades, initial_capital, afternoon_result.equity_curve)}
"""
    print(report)
    print("完了")


if __name__ == "__main__":
    main()
