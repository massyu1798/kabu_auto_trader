"""
バックテスト: オーバーナイト・ギャップ (ONG) 戦略 v1.0 単体実行

ロジック:
  - シグナル当日の引け値でロングエントリー
  - 翌営業日の寄り付きで成行決済
  - config/overnight_config.yaml の設定を使用

実行方法:
    python main_backtest_overnight.py [オプション]

CLI Options:
  --output DIR         出力ディレクトリ (default: カレントディレクトリ)
  --period PERIOD      データ取得期間 (default: 120d)
  --last-trades N      直近N件のトレード明細をコンソール表示 (default: 20)
  --export FILE        トレード明細をCSVで出力
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import yaml

# プロジェクトルートを sys.path に追加（スクリプト直接実行対応）
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backtest.overnight_engine import OvernightGapEngine
from strategy.overnight_gap import generate_ong_signals

# ---------------------------------------------------------------------------
# ロギング設定
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# データ取得ユーティリティ
# ---------------------------------------------------------------------------


def _normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """MultiIndex カラムを単一レベルの小文字カラム名に変換する。"""
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0].lower() for col in data.columns]
    else:
        data.columns = [col.lower() for col in data.columns]
    return data


def load_daily(ticker: str, period: str = "120d") -> Optional[pd.DataFrame]:
    """日足データを Yahoo Finance から取得する。

    Args:
        ticker: Yahoo Finance 形式のティッカー（例: "7203.T"）
        period: yfinance period 文字列（デフォルト "120d"）

    Returns:
        日足 DataFrame、取得失敗時は None
    """
    try:
        raw = yf.download(ticker, period=period, interval="1d", progress=False)
        if raw is None or raw.empty:
            return None
        raw = _normalize_columns(raw)
        raw = raw[["open", "high", "low", "close", "volume"]].copy()
        raw.dropna(inplace=True)
        return raw if not raw.empty else None
    except Exception as exc:
        logger.warning(f"  ⚠ {ticker} 日足取得失敗: {exc}")
        return None


# ---------------------------------------------------------------------------
# レポートユーティリティ
# ---------------------------------------------------------------------------


def calc_max_dd(equity_curve: list) -> float:
    """エクイティカーブから最大ドローダウン(%)を計算する。"""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for e in equity_curve:
        if e > peak:
            peak = e
        if peak > 0:
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd
    return max_dd


def generate_ong_report(result, initial_capital: float) -> str:
    """ONG バックテスト結果のレポート文字列を生成する。"""
    trades = result.trades
    if not trades:
        return "\n(!) トレードが1件も発生しませんでした。条件が厳しすぎる可能性があります。"

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

    max_dd = calc_max_dd(result.equity_curve)

    # Exit理由の集計
    exit_reasons: dict = {}
    for t in trades:
        reason = t.exit_reason
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    exit_summary = "\n".join(
        f"    {reason}: {count}件"
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1])
    )

    # 銘柄別成績サマリ
    ticker_stats: dict = {}
    for t in trades:
        if t.ticker not in ticker_stats:
            ticker_stats[t.ticker] = {"trades": 0, "wins": 0, "pnl": 0.0}
        ticker_stats[t.ticker]["trades"] += 1
        if t.pnl > 0:
            ticker_stats[t.ticker]["wins"] += 1
        ticker_stats[t.ticker]["pnl"] += t.pnl

    # 純損益順でソート
    ticker_lines = []
    for ticker, stats in sorted(ticker_stats.items(), key=lambda x: -x[1]["pnl"]):
        t_count = stats["trades"]
        t_wins = stats["wins"]
        t_wr = t_wins / t_count * 100 if t_count > 0 else 0
        t_pnl = stats["pnl"]
        ticker_lines.append(
            f"    {ticker:<10} {t_count:>4}件  勝率{t_wr:>5.1f}%  損益{t_pnl:>+12,.0f}円"
        )
    ticker_summary = "\n".join(ticker_lines)

    margin = 3_000_000
    margin_ratio = (total_pnl / margin) * 100.0

    report = f"""
============================================================
     オーバーナイト・ギャップ (ONG) バックテストレポート
============================================================

■ 概要
  初期資金:       {initial_capital:>14,.0f} 円
  最終資産:       {equity_final:>14,.0f} 円
  純損益:         {total_pnl:>+14,.0f} 円 ({total_return:+.2f}%)
  信用保証金:     {margin:>14,.0f} 円
  保証金比率:     {margin_ratio:>+13.2f} %  (= 純損益 / 保証金)
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

■ 銘柄別成績
{ticker_summary}
"""
    return report


def save_equity_curve(
    dates: list,
    equity: list,
    label: str,
    initial_capital: float,
    save_path: str,
) -> None:
    """エクイティカーブを PNG に保存する。"""
    if not dates or not equity:
        return
    # 日付ラベルを文字列に変換
    date_labels = [str(d)[:10] for d in dates]
    plt.figure(figsize=(14, 6))
    plt.plot(date_labels, equity, label=label, color="steelblue", linewidth=1.5)
    plt.axhline(y=initial_capital, color="red", linestyle="--", alpha=0.5, label="Initial Capital")
    plt.title(f"{label} — Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity (JPY)")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"エクイティカーブ保存: {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="オーバーナイト・ギャップ (ONG) 戦略 単体バックテスト"
    )
    parser.add_argument(
        "--output",
        default=".",
        help="出力ディレクトリ（デフォルト: カレントディレクトリ）",
    )
    parser.add_argument(
        "--period",
        default="120d",
        help="データ取得期間 (デフォルト: 120d)",
    )
    parser.add_argument(
        "--last-trades",
        type=int,
        default=20,
        metavar="N",
        help="直近N件のトレード明細をコンソール表示 (デフォルト: 20)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        metavar="FILE",
        help="トレード明細をCSVで出力",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------


def main() -> None:
    """ONG 単体バックテストを実行し、結果を出力する。"""
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  オーバーナイト・ギャップ (ONG) バックテスト v1.0")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 設定読み込み
    # ------------------------------------------------------------------
    config_path = os.path.join(_ROOT, "config", "overnight_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ong_tickers: list = config.get("tickers", [])
    nikkei_etf_code: str = config.get("nikkei_etf", "1321.T")
    initial_capital: float = float(config.get("global", {}).get("initial_capital", 3_000_000))

    print(f"\n■ 設定読み込み完了")
    print(f"  対象銘柄数:   {len(ong_tickers)}")
    print(f"  日経ETF:      {nikkei_etf_code}")
    print(f"  取得期間:     {args.period}")
    print(f"  初期資金:     {initial_capital:,.0f} 円")

    # ------------------------------------------------------------------
    # 日足データ取得
    # ------------------------------------------------------------------
    all_tickers = list(set(ong_tickers) | {nikkei_etf_code})
    print(f"\n■ 日足データ取得中 ({len(all_tickers)}銘柄)...")

    daily_cache: dict = {}
    for i, ticker in enumerate(sorted(all_tickers), 1):
        print(f"  [{i:3d}/{len(all_tickers)}] {ticker} ...", end=" ")
        df = load_daily(ticker, period=args.period)
        if df is not None:
            daily_cache[ticker] = df
            print(f"OK ({len(df)}行)")
        else:
            print("取得失敗")

    print(f"\n  取得完了: {len(daily_cache)}/{len(all_tickers)} 銘柄")

    # ------------------------------------------------------------------
    # ONG シグナル生成
    # ------------------------------------------------------------------
    print("\n■ ONG シグナル生成中...")
    ong_daily: dict = {}
    for ticker in ong_tickers:
        df = daily_cache.get(ticker)
        if df is not None and not df.empty:
            ong_daily[ticker] = df

    nikkei_etf_df = daily_cache.get(nikkei_etf_code)
    etf_status = "OK" if nikkei_etf_df is not None else "取得失敗"
    print(f"  対象銘柄: {len(ong_daily)}銘柄 / 日経ETF: {etf_status}")

    ong_signals = generate_ong_signals(ong_daily, nikkei_etf_df, config)

    signal_count = sum(
        int(df["ONG_signal"].sum())
        for df in ong_signals.values()
        if "ONG_signal" in df.columns
    )
    print(f"  シグナル総数: {signal_count}")

    # ------------------------------------------------------------------
    # バックテスト実行
    # ------------------------------------------------------------------
    print("\n■ バックテスト実行中...")
    engine = OvernightGapEngine(config_path)
    result = engine.run(ong_signals)

    # ------------------------------------------------------------------
    # レポート出力
    # ------------------------------------------------------------------
    print(generate_ong_report(result, initial_capital))

    # ------------------------------------------------------------------
    # エクイティカーブ保存
    # ------------------------------------------------------------------
    if result.equity_curve:
        eq_path = os.path.join(args.output, "ong_equity.png")
        save_equity_curve(
            result.dates,
            result.equity_curve,
            "Overnight Gap Strategy",
            initial_capital,
            eq_path,
        )
        print(f"■ エクイティカーブ保存: {eq_path}")
    else:
        print("■ トレードなし: エクイティカーブの保存をスキップしました。")

    # ------------------------------------------------------------------
    # 直近N件のトレード明細表示
    # ------------------------------------------------------------------
    if result.trades:
        n = args.last_trades
        recent = result.trades[-n:]
        print(f"\n■ 直近 {min(n, len(result.trades))} 件のトレード明細")
        print(f"  {'日付':<12} {'銘柄':<10} {'エントリー':>10} {'エグジット':>10} {'株数':>6} {'損益':>12} {'Exit理由'}")
        print("  " + "-" * 80)
        for t in recent:
            entry_date = str(t.entry_date)[:10]
            print(
                f"  {entry_date:<12} {t.ticker:<10} "
                f"{t.entry_price:>10,.1f} {t.exit_price:>10,.1f} "
                f"{t.size:>6} {t.pnl:>+12,.0f}円  {t.exit_reason}"
            )

    # ------------------------------------------------------------------
    # CSV エクスポート
    # ------------------------------------------------------------------
    if args.export and result.trades:
        rows = []
        for t in result.trades:
            rows.append({
                "entry_date": str(t.entry_date)[:10],
                "exit_date": str(t.exit_date)[:10],
                "ticker": t.ticker,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "size": t.size,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "entry_reason": t.entry_reason,
                "exit_reason": t.exit_reason,
            })
        df_export = pd.DataFrame(rows)
        df_export.to_csv(args.export, index=False, encoding="utf-8-sig")
        print(f"\n■ トレード明細CSV出力: {args.export}")

    print("\n完了\n")


if __name__ == "__main__":
    main()
