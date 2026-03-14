"""
バックテスト実行スクリプト: ペア平均回帰戦略

実行方法:
    python main_backtest_pair_meanrev.py [--mode immediate|delayed] [--output <dir>]

データ取得:
  - 5 分足: Yahoo Finance (period="60d")
  - 日足:   Yahoo Finance (period="1y")
  - TOPIX 代替: 1306.T (TOPIX 連動 ETF)

出力:
  - コンソール: 日別シグナル、PnL、最終レポート
  - <output>/pair_meanrev_equity.png  : equity curve PNG
  - <output>/pair_meanrev_trades.csv  : トレード一覧 CSV
  - <output>/pair_meanrev_daily_pnl.csv : 日別 PnL CSV
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from typing import Optional

import matplotlib
# Use the non-interactive Agg backend so the script works in headless environments
# (servers, CI pipelines) that have no display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# プロジェクトルートを sys.path に追加（スクリプト直接実行対応）
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backtest.pair_meanrev_engine import (
    PairMeanRevBacktestEngine,
    PairBacktestResult,
    Side,
)
from strategy.pair_mean_reversion import UNIVERSE

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
# 定数
# ---------------------------------------------------------------------------

CONFIG_PATH = "config/pair_meanrev_config.yaml"
TOPIX_TICKER = "1306.T"

# ユニバース + TOPIX ETF の全 ticker リスト
ALL_TICKERS = list(UNIVERSE.keys()) + [TOPIX_TICKER]


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


def _to_jst(df: pd.DataFrame) -> pd.DataFrame:
    """インデックスを Asia/Tokyo に変換する。"""
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Tokyo")
    elif str(df.index.tz) != "Asia/Tokyo":
        df.index = df.index.tz_convert("Asia/Tokyo")
    return df


def download_intraday(ticker: str) -> Optional[pd.DataFrame]:
    """5 分足データを取得する（最大 60 日分）。

    Args:
        ticker: Yahoo Finance 形式のティッカー（例: "7203.T"）

    Returns:
        JST タイムゾーン付き 5 分足 DataFrame、取得失敗時は None
    """
    try:
        raw = yf.download(ticker, period="60d", interval="5m", progress=False)
        if raw is None or raw.empty:
            return None
        raw = _normalize_columns(raw)
        raw = raw[["open", "high", "low", "close", "volume"]].copy()
        raw.dropna(inplace=True)
        raw = _to_jst(raw)
        # 取引時間（9:00〜15:30 JST）のみ残す
        raw = raw[
            (raw.index.hour >= 9)
            & (
                (raw.index.hour < 15)
                | ((raw.index.hour == 15) & (raw.index.minute <= 30))
            )
        ]
        return raw if not raw.empty else None
    except Exception as exc:
        logger.warning(f"  ⚠ {ticker} 5分足取得失敗: {exc}")
        return None


def download_daily(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """日足データを取得する。

    Args:
        ticker: Yahoo Finance 形式のティッカー
        period: yfinance period 文字列（デフォルト "1y"）

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
# データ読み込み
# ---------------------------------------------------------------------------


def load_all_data() -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """全銘柄の 5 分足・日足データを取得する。

    Returns:
        (intraday_data, daily_data, topix_intraday, topix_daily)
        - intraday_data: 銘柄→5 分足 DataFrame の辞書
        - daily_data:    銘柄→日足 DataFrame の辞書
        - topix_intraday: TOPIX ETF の 5 分足
        - topix_daily:    TOPIX ETF の日足
    """
    print("=" * 60)
    print("  データ取得中 ...")
    print("=" * 60)

    intraday_data: dict[str, pd.DataFrame] = {}
    daily_data: dict[str, pd.DataFrame] = {}

    total = len(ALL_TICKERS)
    for i, ticker in enumerate(ALL_TICKERS, 1):
        print(f"  [{i:2d}/{total}] {ticker} ...", end=" ")

        df_5m = download_intraday(ticker)
        df_1d = download_daily(ticker)

        if df_5m is not None:
            if ticker != TOPIX_TICKER:
                intraday_data[ticker] = df_5m
            status_5m = f"5m:{len(df_5m)}行"
        else:
            status_5m = "5m:なし"

        if df_1d is not None:
            daily_data[ticker] = df_1d
            status_1d = f"1d:{len(df_1d)}行"
        else:
            status_1d = "1d:なし"

        print(f"{status_5m}  {status_1d}")

    # TOPIX データを分離
    _topix_5m = download_intraday(TOPIX_TICKER)
    topix_intraday = _topix_5m if _topix_5m is not None else pd.DataFrame()

    _topix_1d = daily_data.get(TOPIX_TICKER)
    topix_daily = _topix_1d if _topix_1d is not None else pd.DataFrame()

    print(
        f"\n  取得完了: 5分足={len(intraday_data)}銘柄  "
        f"日足={len(daily_data)}銘柄 "
        f"(TOPIX 5m: {'あり' if not topix_intraday.empty else 'なし'})\n"
    )
    return intraday_data, daily_data, topix_intraday, topix_daily


# ---------------------------------------------------------------------------
# 出力ユーティリティ
# ---------------------------------------------------------------------------


def save_equity_curve(
    result: PairBacktestResult,
    initial_capital: float,
    save_path: str,
) -> None:
    """エクイティカーブを PNG に保存する。

    Args:
        result:          バックテスト結果
        initial_capital: 初期資本
        save_path:       保存先ファイルパス
    """
    if not result.dates or not result.equity_curve:
        logger.warning("エクイティカーブのデータがありません")
        return

    plt.figure(figsize=(14, 6))
    plt.plot(result.dates, result.equity_curve, label="Equity", color="steelblue", linewidth=1.5)
    plt.axhline(y=initial_capital, color="red", linestyle="--", alpha=0.5, label="Initial Capital")
    plt.title("Pair Mean Reversion Strategy — Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity (JPY)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"エクイティカーブ保存: {save_path}")


def save_trades_csv(result: PairBacktestResult, save_path: str) -> None:
    """トレード一覧を CSV に保存する。

    Args:
        result:    バックテスト結果
        save_path: 保存先 CSV パス
    """
    if not result.trades:
        logger.warning("トレードデータがありません")
        return

    rows = []
    for t in result.trades:
        rows.append(
            {
                "pair_id": t.pair_id,
                "trade_date": t.trade_date,
                "ticker": t.ticker,
                "side": t.side.value,
                "sector": t.sector,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "size": t.size,
                "pnl": round(t.pnl, 0),
                "pnl_pct": round(t.pnl_pct, 3),
                "entry_date": str(t.entry_date),
                "exit_date": str(t.exit_date),
                "entry_reason": t.entry_reason,
                "exit_reason": t.exit_reason,
            }
        )
    pd.DataFrame(rows).to_csv(save_path, index=False, encoding="utf-8-sig")
    logger.info(f"トレード CSV 保存: {save_path}")


def save_daily_pnl_csv(result: PairBacktestResult, save_path: str) -> None:
    """日別 PnL を CSV に保存する。

    Args:
        result:    バックテスト結果
        save_path: 保存先 CSV パス
    """
    if not result.daily_pnl:
        logger.warning("日別 PnL データがありません")
        return

    rows = [{"date": d, "pnl": round(v, 0)} for d, v in sorted(result.daily_pnl.items())]
    pd.DataFrame(rows).to_csv(save_path, index=False, encoding="utf-8-sig")
    logger.info(f"日別 PnL CSV 保存: {save_path}")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="ペア平均回帰戦略バックテスト"
    )
    parser.add_argument(
        "--mode",
        choices=["immediate", "delayed"],
        default=None,
        help="exit_mode の上書き（config より優先）",
    )
    parser.add_argument(
        "--output",
        default=".",
        help="出力ディレクトリ（デフォルト: カレントディレクトリ）",
    )
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help=f"設定ファイルパス（デフォルト: {CONFIG_PATH}）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="デバッグログを有効化",
    )
    return parser.parse_args()


def main() -> None:
    """バックテストを実行し、結果を出力する。"""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    os.makedirs(args.output, exist_ok=True)

    print("\n" + "=" * 60)
    print("  ペア平均回帰戦略 バックテスト")
    print("  (前場終了時相対過熱・過冷えスコア → 後場寄り平均回帰)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # データ取得
    # ------------------------------------------------------------------
    intraday_data, daily_data, topix_intraday, topix_daily = load_all_data()

    if not intraday_data:
        logger.error("5分足データが取得できませんでした。終了します。")
        print("\n⚠️ 5分足データが取得できませんでした。終了します。")
        sys.exit(1)

    if topix_intraday.empty:
        print("\n⚠️ TOPIX ETF (1306.T) の 5 分足データが取得できません。")
        print("   TOPIX フィルタは無効化されます。")

    # ------------------------------------------------------------------
    # エンジン初期化（exit_mode の上書きオプション対応）
    # ------------------------------------------------------------------
    engine = PairMeanRevBacktestEngine(args.config)

    if args.mode is not None:
        engine.exit_mode = args.mode
        logger.info(f"exit_mode を '{args.mode}' に上書き")

    print(
        f"\n■ バックテスト設定\n"
        f"  初期資金:       {engine.initial_capital:>12,.0f} 円\n"
        f"  exit_mode:      {engine.exit_mode}\n"
        f"  SL:             {engine.stop_loss_pct * 100:.1f}%  "
        f"TP: {engine.take_profit_pct * 100:.1f}%\n"
        f"  risk_per_pos:   {engine.risk_per_position * 100:.0f}%\n"
    )

    # ------------------------------------------------------------------
    # バックテスト実行
    # ------------------------------------------------------------------
    print("■ バックテスト実行中 ...\n")
    result = engine.run(intraday_data, daily_data, topix_intraday, topix_daily)

    # ------------------------------------------------------------------
    # レポート出力
    # ------------------------------------------------------------------
    print(engine.generate_report(result))

    if not result.trades:
        print("トレードが発生しませんでした。設定・データを確認してください。")
        sys.exit(0)

    # ------------------------------------------------------------------
    # ファイル出力
    # ------------------------------------------------------------------
    eq_path = os.path.join(args.output, "pair_meanrev_equity.png")
    trades_path = os.path.join(args.output, "pair_meanrev_trades.csv")
    dpnl_path = os.path.join(args.output, "pair_meanrev_daily_pnl.csv")

    save_equity_curve(result, engine.initial_capital, eq_path)
    save_trades_csv(result, trades_path)
    save_daily_pnl_csv(result, dpnl_path)

    print(f"\n■ 出力ファイル")
    print(f"  エクイティカーブ: {eq_path}")
    print(f"  トレード一覧:     {trades_path}")
    print(f"  日別PnL:          {dpnl_path}")
    print("\n完了\n")


if __name__ == "__main__":
    main()
