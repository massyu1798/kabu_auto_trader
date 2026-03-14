"""
ペア戦略バックテスト詳細レポート生成モジュール

generate_report() を補完するスタンドアロンのレポーターモジュール。
より詳細な分析・可視化機能を提供する。
"""

from __future__ import annotations

import logging
import statistics
from typing import Optional

import pandas as pd

# プロジェクトルート直下から実行する場合（`python backtest/pair_meanrev_reporter.py`）と
# プロジェクトルートからモジュールとしてインポートする場合（`from backtest.pair_meanrev_reporter import ...`）
# の両方をサポートするためにフォールバックインポートを使用する。
try:
    from backtest.pair_meanrev_engine import PairBacktestResult, PairTrade, Side
    from strategy.pair_mean_reversion import UNIVERSE
except ImportError:
    from pair_meanrev_engine import PairBacktestResult, PairTrade, Side  # type: ignore[no-redef]
    from pair_mean_reversion import UNIVERSE  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


def _calc_max_drawdown(
    equity_curve: list[float],
) -> tuple[float, float]:
    """エクイティカーブから最大ドローダウン（%・金額）を算出する。

    Args:
        equity_curve: エクイティカーブのリスト

    Returns:
        (max_dd_pct, max_dd_amount) のタプル
    """
    if not equity_curve:
        return 0.0, 0.0
    peak = equity_curve[0]
    max_dd_pct = 0.0
    max_dd_amount = 0.0
    for e in equity_curve:
        if e > peak:
            peak = e
        dd_amount = peak - e
        dd_pct = dd_amount / peak * 100.0 if peak > 0 else 0.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_amount = dd_amount
    return max_dd_pct, max_dd_amount


def _calc_sharpe(daily_pnl_vals: list[float]) -> float:
    """日次 PnL から年率 Sharpe 比率を算出する（√252 スケール）。

    Args:
        daily_pnl_vals: 日次 PnL のリスト

    Returns:
        Sharpe 比率（データ不足・標準偏差ゼロ時は 0.0）
    """
    if len(daily_pnl_vals) < 2:
        return 0.0
    mean_d = statistics.mean(daily_pnl_vals)
    std_d = statistics.stdev(daily_pnl_vals)
    if std_d <= 0:
        return 0.0
    return (mean_d / std_d) * (252 ** 0.5)


def generate_full_report(
    result: PairBacktestResult,
    initial_capital: float,
    mode_label: str = "",
) -> str:
    """バックテスト結果の完全詳細レポートを生成する。

    以下の全評価指標を出力する:
        - 基本統計: 勝率・総トレード数・勝ち/負け回数
        - PnL 統計: 総損益・平均利益・平均損失・最大利益・最大損失
        - リスク指標: PF・最大ドローダウン（金額・%）・Sharpe Ratio・Calmar Ratio
        - 日別損益: 平均・標準偏差・最大・最小
        - LONG/SHORT 別寄与: 合計 PnL・各勝率
        - 銘柄別寄与: 銘柄ごとの合計 PnL・取引回数・勝率
        - セクター別寄与: セクターごとの合計 PnL・件数・勝率
        - 月別 PnL: 月ごとの損益集計
        - Exit 理由内訳

    Args:
        result:          バックテスト結果 (PairBacktestResult)
        initial_capital: 初期資本（円）
        mode_label:      モードラベル（"immediate" / "delayed" 等、空文字可）

    Returns:
        整形済みテキストレポート
    """
    trades = result.trades
    ic = initial_capital

    if not trades:
        return "\n⚠️ トレードが1件も発生しませんでした。"

    # ----------------------------------------------------------------
    # 基本統計
    # ----------------------------------------------------------------
    total = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / total * 100.0

    # ----------------------------------------------------------------
    # PnL 統計
    # ----------------------------------------------------------------
    total_pnl = sum(t.pnl for t in trades)
    total_ret = total_pnl / ic * 100.0
    equity_final = ic + total_pnl

    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
    max_win = max((t.pnl for t in trades), default=0.0)
    max_loss = min((t.pnl for t in trades), default=0.0)
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    # ----------------------------------------------------------------
    # リスク指標
    # ----------------------------------------------------------------
    max_dd_pct, max_dd_amount = _calc_max_drawdown(result.equity_curve)
    daily_vals = list(result.daily_pnl.values())
    sharpe = _calc_sharpe(daily_vals)
    calmar = (total_ret / max_dd_pct) if max_dd_pct > 0 else float("inf")

    # ----------------------------------------------------------------
    # 日別損益統計
    # ----------------------------------------------------------------
    daily_mean = statistics.mean(daily_vals) if daily_vals else 0.0
    daily_std = statistics.stdev(daily_vals) if len(daily_vals) >= 2 else 0.0
    daily_max = max(daily_vals) if daily_vals else 0.0
    daily_min = min(daily_vals) if daily_vals else 0.0

    # ----------------------------------------------------------------
    # LONG / SHORT 別寄与
    # ----------------------------------------------------------------
    long_trades = [t for t in trades if t.side == Side.LONG]
    short_trades = [t for t in trades if t.side == Side.SHORT]
    long_wins = [t for t in long_trades if t.pnl > 0]
    short_wins = [t for t in short_trades if t.pnl > 0]
    long_pnl = sum(t.pnl for t in long_trades)
    short_pnl = sum(t.pnl for t in short_trades)
    long_wr = len(long_wins) / len(long_trades) * 100.0 if long_trades else 0.0
    short_wr = len(short_wins) / len(short_trades) * 100.0 if short_trades else 0.0

    # ----------------------------------------------------------------
    # 銘柄別寄与
    # ----------------------------------------------------------------
    ticker_stats: dict[str, dict] = {}
    for t in trades:
        tk = t.ticker
        if tk not in ticker_stats:
            ticker_stats[tk] = {"pnl": 0.0, "count": 0, "wins": 0}
        ticker_stats[tk]["pnl"] += t.pnl
        ticker_stats[tk]["count"] += 1
        if t.pnl > 0:
            ticker_stats[tk]["wins"] += 1

    ticker_lines = []
    for tk, st in sorted(ticker_stats.items(), key=lambda x: -x[1]["pnl"]):
        wr = st["wins"] / st["count"] * 100.0 if st["count"] > 0 else 0.0
        name = UNIVERSE.get(tk, {}).get("name", tk)
        ticker_lines.append(
            f"    {tk:8s} {name:14s}  {st['pnl']:>+10,.0f}円  "
            f"{st['count']:3d}件  勝率{wr:.0f}%"
        )

    # ----------------------------------------------------------------
    # セクター別寄与
    # ----------------------------------------------------------------
    sector_stats: dict[str, dict] = {}
    for t in trades:
        s = t.sector or "不明"
        if s not in sector_stats:
            sector_stats[s] = {"pnl": 0.0, "count": 0, "wins": 0}
        sector_stats[s]["pnl"] += t.pnl
        sector_stats[s]["count"] += 1
        if t.pnl > 0:
            sector_stats[s]["wins"] += 1

    sector_lines = []
    for sec, st in sorted(sector_stats.items(), key=lambda x: -x[1]["pnl"]):
        wr = st["wins"] / st["count"] * 100.0 if st["count"] > 0 else 0.0
        sector_lines.append(
            f"    {sec:12s}  {st['pnl']:>+10,.0f}円  "
            f"{st['count']:3d}件  勝率{wr:.0f}%"
        )

    # ----------------------------------------------------------------
    # 月別 PnL
    # ----------------------------------------------------------------
    monthly_pnl: dict[str, float] = {}
    for t in trades:
        month_key = t.trade_date[:7]  # "YYYY-MM"
        monthly_pnl[month_key] = monthly_pnl.get(month_key, 0.0) + t.pnl
    monthly_lines = "\n".join(
        f"    {month}  {v:>+10,.0f}円"
        for month, v in sorted(monthly_pnl.items())
    )

    # ----------------------------------------------------------------
    # 日別損益（上位 / 下位 5 日）
    # ----------------------------------------------------------------
    dpnl_items = sorted(result.daily_pnl.items(), key=lambda x: x[1])
    best5 = list(reversed(dpnl_items[-5:])) if dpnl_items else []
    worst5 = dpnl_items[:5] if dpnl_items else []
    best_lines = "\n".join(f"    {d}  {v:>+10,.0f}円" for d, v in best5)
    worst_lines = "\n".join(f"    {d}  {v:>+10,.0f}円" for d, v in worst5)

    # ----------------------------------------------------------------
    # Exit 理由内訳
    # ----------------------------------------------------------------
    exit_reasons: dict[str, int] = {}
    for t in trades:
        key = t.exit_reason.split("(")[0]
        exit_reasons[key] = exit_reasons.get(key, 0) + 1
    exit_lines = "\n".join(
        f"    {r}: {c}件"
        for r, c in sorted(exit_reasons.items(), key=lambda x: -x[1])
    )

    # ----------------------------------------------------------------
    # モードラベル
    # ----------------------------------------------------------------
    mode_str = f" [{mode_label}]" if mode_label else ""

    report = f"""
============================================================
     ペアモメンタム継続戦略 バックテストレポート{mode_str}
============================================================

■ 概要
  初期資金:          {ic:>15,.0f} 円
  最終資産:          {equity_final:>15,.0f} 円
  純損益:            {total_pnl:>+15,.0f} 円 ({total_ret:+.2f}%)
  最大DD:            {max_dd_pct:>14.2f} %  ({max_dd_amount:>+12,.0f} 円)
  Sharpe 比率:       {sharpe:>14.3f}
  Calmar 比率:       {calmar:>14.3f}

■ トレード統計
  総トレード数:      {total:>5d}
  勝ちトレード:      {len(wins):>5d} ({win_rate:.1f}%)
  負けトレード:      {len(losses):>5d} ({100 - win_rate:.1f}%)
  平均利益:          {avg_win:>+15,.0f} 円
  平均損失:          {avg_loss:>+15,.0f} 円
  最大利益:          {max_win:>+15,.0f} 円
  最大損失:          {max_loss:>+15,.0f} 円
  PF:                {pf:.3f}

■ 日別損益統計
  平均:              {daily_mean:>+15,.0f} 円
  標準偏差:          {daily_std:>15,.0f} 円
  最大:              {daily_max:>+15,.0f} 円
  最小:              {daily_min:>+15,.0f} 円

■ LONG / SHORT 内訳
  LONG:   {len(long_trades):>4d}件  勝ち{len(long_wins):>3d}件  勝率{long_wr:5.1f}%  PnL={long_pnl:>+12,.0f} 円
  SHORT:  {len(short_trades):>4d}件  勝ち{len(short_wins):>3d}件  勝率{short_wr:5.1f}%  PnL={short_pnl:>+12,.0f} 円

■ 銘柄別寄与
{chr(10).join(ticker_lines) if ticker_lines else "  データなし"}

■ セクター別成績
{chr(10).join(sector_lines) if sector_lines else "  データなし"}

■ 月別 PnL
{monthly_lines if monthly_lines else "  データなし"}

■ 日別損益 Top 5
{best_lines if best_lines else "  データなし"}

■ 日別損益 Worst 5
{worst_lines if worst_lines else "  データなし"}

■ Exit 理由内訳
{exit_lines if exit_lines else "  データなし"}
============================================================
"""
    return report


def compare_reports(
    result_immediate: PairBacktestResult,
    result_delayed: PairBacktestResult,
    initial_capital: float,
) -> str:
    """immediate / delayed 両モードの結果を並べて比較するレポートを生成する。

    Args:
        result_immediate: immediate モードのバックテスト結果
        result_delayed:   delayed モードのバックテスト結果
        initial_capital:  初期資本（円）

    Returns:
        整形済み比較レポート文字列
    """
    def _summary(result: PairBacktestResult) -> dict:
        """結果から主要指標の辞書を生成する内部ヘルパー。"""
        trades = result.trades
        ic = initial_capital
        if not trades:
            return {
                "total": 0, "win_rate": 0.0, "total_pnl": 0.0,
                "total_ret": 0.0, "pf": 0.0, "max_dd_pct": 0.0,
                "sharpe": 0.0, "calmar": 0.0,
            }
        total = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        win_rate = len(wins) / total * 100.0
        total_pnl = sum(t.pnl for t in trades)
        total_ret = total_pnl / ic * 100.0
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in [t for t in trades if t.pnl <= 0]))
        pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
        max_dd_pct, _ = _calc_max_drawdown(result.equity_curve)
        daily_vals = list(result.daily_pnl.values())
        sharpe = _calc_sharpe(daily_vals)
        calmar = (total_ret / max_dd_pct) if max_dd_pct > 0 else float("inf")
        return {
            "total": total, "win_rate": win_rate, "total_pnl": total_pnl,
            "total_ret": total_ret, "pf": pf, "max_dd_pct": max_dd_pct,
            "sharpe": sharpe, "calmar": calmar,
        }

    imm = _summary(result_immediate)
    dly = _summary(result_delayed)

    report = f"""
============================================================
    モード比較レポート: immediate vs delayed
============================================================

  指標               immediate          delayed
  -----------------------------------------------
  総トレード数       {imm['total']:>10d}  {dly['total']:>10d}
  勝率               {imm['win_rate']:>9.1f}%  {dly['win_rate']:>9.1f}%
  純損益             {imm['total_pnl']:>+10,.0f}円  {dly['total_pnl']:>+10,.0f}円
  リターン           {imm['total_ret']:>+9.2f}%  {dly['total_ret']:>+9.2f}%
  PF                 {imm['pf']:>10.3f}  {dly['pf']:>10.3f}
  最大DD             {imm['max_dd_pct']:>9.2f}%  {dly['max_dd_pct']:>9.2f}%
  Sharpe             {imm['sharpe']:>10.3f}  {dly['sharpe']:>10.3f}
  Calmar             {imm['calmar']:>10.3f}  {dly['calmar']:>10.3f}

============================================================
"""
    return report
