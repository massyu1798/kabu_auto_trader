"""バックテスト結果のレポート生成（エラー耐性強化版）"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate
try:
    from backtest.engine import BacktestResult, Side
except ImportError:
    # 実行環境によってパスが変わる場合の対策
    from engine import BacktestResult, Side

def generate_report(result: BacktestResult, initial_capital: float) -> str:
    trades = result.trades
    if not trades:
        return "\n⚠️ トレードが1件も発生しませんでした。条件が厳しすぎる可能性があります。"

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
    profit_factor = (total_win_amt / total_loss_amt) if total_loss_amt != 0 else float('inf')

    # 最大ドローダウン計算
    equity = result.equity_curve
    max_dd = 0
    if equity:
        peak = equity[0]
        for e in equity:
            if e > peak: peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd: max_dd = dd

    report = f"""
============================================================
           バックテストレポート (v12 安定版)
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
"""
    return report

def plot_equity_curve(result: BacktestResult, initial_capital: float, save_path: str = "equity_curve.png"):
    if not result.dates or not result.equity_curve:
        print("⚠️ データがないためグラフを作成できません。")
        return
        
    plt.figure(figsize=(12, 6))
    plt.plot(result.dates, result.equity_curve, label="Equity")
    plt.axhline(y=initial_capital, color="red", linestyle="--", alpha=0.5)
    plt.title("Backtest Equity Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()