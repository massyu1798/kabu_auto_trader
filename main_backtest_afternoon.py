"""
バックテスト: 午後リバーサル戦略
- 午前中に動きすぎた銘柄のVWAP回帰を狙う
- エントリー: 12:30〜14:00
- エグジット: VWAP回帰 / SL / 14:50強制決済
"""

import pandas as pd
import yfinance as yf
import yaml
import pandas_ta as ta
from backtest.afternoon_engine import AfternoonBacktestEngine, BacktestResult, Side
from backtest.screener import screen_stocks
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
        # 取引時間フィルター
        data = data[
            (data.index.hour >= 9)
            & ((data.index.hour < 15) | ((data.index.hour == 15) & (data.index.minute <= 25)))
        ]
        return data
    except Exception as e:
        print(f"  ❌ {ticker} 取得エラー: {e}")
        return None


def generate_report(result: BacktestResult, initial_capital: float) -> str:
    trades = result.trades
    if not trades:
        return "\n⚠️ トレードが1件も発生しま��んでした。条件が厳しすぎる可能性があります。"

    total_trades = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / total_trades * 100

    total_pnl = sum(t.pnl for t in trades)
    total_return = (total_pnl / initial_capital) * 100
    equity_final = initial_capital + total_pnl

    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

    total_win_amt = sum(t.pnl for t in wins)
    total_loss_amt = abs(sum(t.pnl for t in losses))
    profit_factor = (total_win_amt / total_loss_amt) if total_loss_amt != 0 else float("inf")

    # 最大ドローダウン計算
    equity = result.equity_curve
    max_dd = 0
    if equity:
        peak = equity[0]
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd

    # Exit理由の集計
    exit_reasons = {}
    for t in trades:
        reason = t.exit_reason.split(" ")[0]  # 括弧前の部分
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    exit_summary = "\n".join(f"    {reason}: {count}件" for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]))

    report = f"""
============================================================
     午後リバーサル バックテストレポート
============================================================

■ 概要
  初期資金:       {initial_capital:>14,.0f} 円
  最終資産:       {equity_final:>14,.0f} 円
  純損益:         {total_pnl:>+14,.0f} 円 ({total_return:+.2f}%)
  最大DD:         {max_dd:>13.2f} %

■ トレード統計
  総トレード数:   {total_trades}
  勝ちトレード:   {len(wins)} ({win_rate:.1f}%)
  負けトレード:   {len(losses)} ({100 - win_rate:.1f}%)
  平均利益:       {avg_win:>+14,.0f} 円
  平均損失:       {avg_loss:>+14,.0f} 円
  PF:             {profit_factor:.2f}

■ Exit理由内訳
{exit_summary}
"""
    return report


def main():
    print("=" * 60)
    print("  🔄 午後リバー���ル バックテスト")
    print("=" * 60)

    with open("config/afternoon_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("\n■ 銘柄スクリーニング...")
    selected = screen_stocks(config)
    tickers = [s["ticker"] for s in selected]

    print(f"\n■ 解析中（午後リバーサルシグナル生成）...")
    reversal_engine = AfternoonReversalEngine("config/afternoon_config.yaml")
    signals_dict = {}

    for ticker in tickers:
        df_5m = load_intraday(ticker)
        if df_5m is None or len(df_5m) < 20:
            continue

        signals_df = reversal_engine.generate_signals(df_5m)
        signals_dict[ticker] = signals_df
        print(".", end="", flush=True)

    print(f"\n\n■ バックテスト実行（{len(signals_dict)}銘柄）...")
    engine = AfternoonBacktestEngine("config/afternoon_config.yaml")
    result = engine.run(signals_dict)

    print(generate_report(result, engine.initial_capital))
    print("\n✅ 完了")


if __name__ == "__main__":
    main()
