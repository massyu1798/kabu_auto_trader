"""
合体バックテスト: 午前 v12.4 + 午後リバーサル v1.2 + オーバーナイト・ギャップ v1.0 + シンプル順張り(値幅 ≥ 3.0%)

4戦略を各々独立にバックテストし、日別PnLを合算して総合エクイティカーブ・総合レポートを出力する。

実行方法:
    python main_backtest_all_combined.py [--output <dir>]

CLI Options:
  --output DIR         出力ディレクトリ (default: カレントディレクトリ)
  --last-days N        直近N営業日の日次レポート (default: 7)
  --export FILE        トレード明細をCSV/JSON出力 (拡張子で判定)
  --export-daily FILE  日次集計をCSV/JSON出力
  --print-latest       最新日のAM/PM/ONG/SM別損益をコンソール表示
  --no-summary         総合サマリ出力を抑制
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
import pandas_ta as ta

# プロジェクトルートを sys.path に追加（スクリプト直接実行対応）
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backtest.engine import BacktestEngine
from backtest.afternoon_engine import AfternoonBacktestEngine
from backtest.overnight_engine import OvernightGapEngine
from backtest.simple_momentum_engine import SimpleMomentumBacktestEngine
from backtest.screener import screen_stocks
from backtest.trade_export import (
    build_trades_df,
    build_daily_pnl,
    get_latest_day_summary,
    export_trades_csv,
    export_trades_json,
    export_daily_csv,
    export_daily_json,
    print_daily_table,
    print_latest_day,
)
from strategy.ensemble import EnsembleEngine
from strategy.afternoon_reversal import AfternoonReversalEngine
from strategy.overnight_gap import generate_ong_signals
from strategy.universe import UNIVERSE

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


def _to_jst(df: pd.DataFrame) -> pd.DataFrame:
    """インデックスを Asia/Tokyo に変換する。"""
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Tokyo")
    elif str(df.index.tz) != "Asia/Tokyo":
        df.index = df.index.tz_convert("Asia/Tokyo")
    return df


def load_intraday_5m(ticker: str, with_vwap: bool = False) -> Optional[pd.DataFrame]:
    """5分足データを取得する（最大60日分）。

    Args:
        ticker:    Yahoo Finance 形式のティッカー（例: "7203.T"）
        with_vwap: True の場合 VWAP カラムを追加する

    Returns:
        JST タイムゾーン付き 5分足 DataFrame、取得失敗時は None
    """
    try:
        raw = yf.download(ticker, period="60d", interval="5m", progress=False)
        if raw is None or raw.empty:
            return None
        raw = _normalize_columns(raw)
        raw = raw[["open", "high", "low", "close", "volume"]].copy()
        raw.dropna(inplace=True)
        raw = _to_jst(raw)
        if with_vwap:
            raw["vwap"] = ta.vwap(raw["high"], raw["low"], raw["close"], raw["volume"])
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


def load_daily(ticker: str, period: str = "120d") -> Optional[pd.DataFrame]:
    """日足データを取得する。

    Args:
        ticker: Yahoo Finance 形式のティッカー
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


def format_report_section(title: str, trades: list, initial_capital: float, equity_curve: list) -> str:
    """戦略セクション別レポート文字列を生成する。"""
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

    max_dd = 0.0
    if equity_curve:
        peak = equity_curve[0]
        for e in equity_curve:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd

    margin = 3_000_000
    margin_ratio = (total_pnl / margin) * 100.0

    return f"""
  [{title}]
    純損益:         {total_pnl:>+14,.0f} 円 ({total_return:+.2f}%)
    信用保証金:     {margin:>14,.0f} 円
    保証金比率:     {margin_ratio:>+13.2f} %  (= 純損益 / 保証金)
    最大DD:         {max_dd:>13.2f} %
    トレード数:     {total_trades} (勝ち{len(wins)} / 負け{len(losses)})
    勝率:           {win_rate:.1f}%
    平均利益:       {avg_win:>+14,.0f} 円
    平均損失:       {avg_loss:>+14,.0f} 円
    PF:             {pf:.2f}
"""


def calc_dd(equity_curve: list) -> float:
    """エクイティカーブから最大ドローダウン(%)を計算する。"""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for e in equity_curve:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _detect_format(filepath: str) -> str:
    """ファイル拡張子からフォーマットを判定する。"""
    return "json" if filepath.lower().endswith(".json") else "csv"


def calc_daily_bias(daily_df: pd.DataFrame, config: dict) -> dict:
    """日足データから日次バイアス（BULL/BEAR/NEUTRAL）を計算する。"""
    dc = config["daily_bias"]
    df = daily_df.copy()
    df["ema_s"] = ta.ema(df["close"], length=dc["ema_short"])
    df["ema_l"] = ta.ema(df["close"], length=dc["ema_long"])
    bias: dict = {}
    for idx, row in df.iterrows():
        date = idx.date() if hasattr(idx, "date") else idx
        if pd.isna(row["ema_s"]) or pd.isna(row["ema_l"]):
            bias[date] = "NEUTRAL"
        elif row["ema_s"] > row["ema_l"]:
            bias[date] = "BULL"
        else:
            bias[date] = "BEAR"
    return bias


def _is_morning_session(timestamp) -> bool:
    """午前v12.4: 12:00までのシグナルのみ有効にするフィルター"""
    if not hasattr(timestamp, "hour"):
        return True
    t = timestamp.hour * 100 + timestamp.minute
    return 900 <= t <= 1200


def apply_v12_filters(signals_df: pd.DataFrame, daily_bias: dict) -> pd.DataFrame:
    """v12バイアスフィルター + 時間帯フィルター + VWAPフィルターを適用する。"""
    result = signals_df.copy()
    for idx in result.index:
        date = idx.date() if hasattr(idx, "date") else idx

        # 日足バイアスフィルター
        b = daily_bias.get(date, "NEUTRAL")
        if b == "BEAR":
            result.loc[idx, "final_signal"] = "HOLD"
            continue

        # 時間帯フィルター（9:05〜11:00 以外はHOLD）
        t = idx.hour * 100 + idx.minute
        if not (905 <= t <= 1100):
            result.loc[idx, "final_signal"] = "HOLD"
            continue

        # VWAPフィルター（VWAP以下での買いはHOLD）
        if result.loc[idx, "final_signal"] == "BUY":
            if "vwap" in result.columns and not pd.isna(result.loc[idx, "vwap"]):
                if result.loc[idx, "close"] < result.loc[idx, "vwap"]:
                    result.loc[idx, "final_signal"] = "HOLD"

    return result


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
    plt.figure(figsize=(14, 6))
    plt.plot(dates, equity, label=label, color="steelblue", linewidth=1.5)
    plt.axhline(y=initial_capital, color="red", linestyle="--", alpha=0.5, label="Initial Capital")
    plt.title(f"{label} — Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity (JPY)")
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
        description=(
            "合体バックテスト: 午前 v12.4 + 午後リバーサル v1.2 + ONG v1.0 "
            "+ シンプル順張り(値幅 ≥ 3.0%)"
        )
    )
    parser.add_argument(
        "--output",
        default=".",
        help="出力ディレクトリ（デフォルト: カレントディレクトリ）",
    )
    parser.add_argument(
        "--last-days", type=int, default=7,
        help="直近N営業日の日次レポートを表示 (default: 7)",
    )
    parser.add_argument(
        "--export", type=str, default=None, metavar="FILE",
        help="トレード明細をCSV/JSONで出力 (拡張子で判定)",
    )
    parser.add_argument(
        "--export-daily", type=str, default=None, metavar="FILE",
        help="日次集計をCSV/JSONで出力",
    )
    parser.add_argument(
        "--print-latest", action="store_true",
        help="最新日（データ上の最終日）の戦略別損益を表示",
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="総合サマリ出力を抑制",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------


def main() -> None:
    """4戦略の合体バックテストを実行し、結果を出力する。"""
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  午前v12.4 + 午後リバーサルv1.2 + ONGv1.0")
    print("  + シンプル順張り(値幅 ≥ 3.0%) 合体バックテスト")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 設定読み込み
    # ------------------------------------------------------------------
    with open("config/strategy_config.yaml", "r", encoding="utf-8") as f:
        morning_config = yaml.safe_load(f)
    with open("config/afternoon_config.yaml", "r", encoding="utf-8") as f:
        afternoon_config = yaml.safe_load(f)
    with open("config/overnight_config.yaml", "r", encoding="utf-8") as f:
        overnight_config = yaml.safe_load(f)

    initial_capital = morning_config["global"]["initial_capital"]
    ong_capital = overnight_config["global"]["initial_capital"]

    # シンプル順張りエンジンの初期化（設定読み込みのみ）
    sm_engine = SimpleMomentumBacktestEngine("config/simple_momentum_config.yaml")
    sm_capital = sm_engine.initial_capital

    # ------------------------------------------------------------------
    # 銘柄スクリーニング（午前/午後共通）
    # ------------------------------------------------------------------
    print("\n■ 銘柄スクリーニング（午前/午後）...")
    selected = screen_stocks(morning_config)
    am_pm_tickers = [s["ticker"] for s in selected]
    print(f"  -> {len(am_pm_tickers)}銘柄を選定")

    # シンプル順張りのユニバース
    sm_tickers = list(UNIVERSE.keys())

    # ONG 銘柄
    ong_tickers = overnight_config.get("tickers", [])
    nikkei_etf_code = overnight_config.get("nikkei_etf", "1321.T")

    # 全ユニーク銘柄をまとめてダウンロード対象とする（重複ダウンロード回避）
    all_tickers_set: set[str] = (
        set(am_pm_tickers) | set(sm_tickers) | set(ong_tickers) | {nikkei_etf_code}
    )
    all_tickers_list = sorted(all_tickers_set)

    # ------------------------------------------------------------------
    # 全銘柄データ取得（共有）
    # ------------------------------------------------------------------
    print(f"\n■ データ取得中 ({len(all_tickers_list)}銘柄)...")

    intraday_cache: dict[str, pd.DataFrame] = {}   # with VWAP (午前/午後用)
    intraday_plain: dict[str, pd.DataFrame] = {}   # without VWAP (SM用)
    daily_cache: dict[str, pd.DataFrame] = {}

    for i, ticker in enumerate(all_tickers_list, 1):
        print(f"  [{i:3d}/{len(all_tickers_list)}] {ticker} ...", end=" ")
        df_5m_vwap = load_intraday_5m(ticker, with_vwap=True)
        df_5m = None
        if df_5m_vwap is not None:
            df_5m_vwap_dropped = df_5m_vwap.drop(columns=["vwap"], errors="ignore")
            df_5m = df_5m_vwap_dropped
        df_1d = load_daily(ticker)
        if df_5m_vwap is not None:
            intraday_cache[ticker] = df_5m_vwap
            intraday_plain[ticker] = df_5m_vwap.drop(columns=["vwap"], errors="ignore")
            status_5m = f"5m:{len(df_5m_vwap)}行"
        else:
            status_5m = "5m:なし"
        if df_1d is not None:
            daily_cache[ticker] = df_1d
            status_1d = f"1d:{len(df_1d)}行"
        else:
            status_1d = "1d:なし"
        print(f"{status_5m}  {status_1d}")

    print(
        f"\n  取得完了: 5分足={len(intraday_cache)}銘柄  "
        f"日足={len(daily_cache)}銘柄\n"
    )

    # ------------------------------------------------------------------
    # 午前シグナル生成
    # ------------------------------------------------------------------
    print("\n■ 午前シグナル生成中...")
    ensemble = EnsembleEngine("config/strategy_config.yaml")
    morning_signals: dict = {}

    for ticker in am_pm_tickers:
        df_5m = intraday_cache.get(ticker)
        df_daily = daily_cache.get(ticker)
        if df_5m is None or len(df_5m) < 20:
            continue
        if df_daily is not None:
            bias = calc_daily_bias(df_daily, morning_config)
            signals_df = ensemble.generate_ensemble_signals(df_5m)
            signals_df = apply_v12_filters(signals_df, bias)
            morning_signals[ticker] = signals_df
        print(".", end="", flush=True)

    print(f"\n  -> 午前: {len(morning_signals)}銘柄")

    # ------------------------------------------------------------------
    # 午後シグナル生成
    # ------------------------------------------------------------------
    print("\n■ 午後シグナル生成中...")
    reversal_engine = AfternoonReversalEngine("config/afternoon_config.yaml")
    afternoon_signals: dict = {}

    for ticker in am_pm_tickers:
        df_5m = intraday_cache.get(ticker)
        if df_5m is None or len(df_5m) < 20:
            continue
        df_plain = df_5m.drop(columns=["vwap"], errors="ignore")
        afternoon_df = reversal_engine.generate_signals(df_plain)
        afternoon_signals[ticker] = afternoon_df
        print(".", end="", flush=True)

    print(f"\n  -> 午後: {len(afternoon_signals)}銘柄")

    # ------------------------------------------------------------------
    # ONG シグナル生成
    # ------------------------------------------------------------------
    print("\n■ ONG データ/シグナル生成中...")
    ong_daily: dict = {}
    for ticker in ong_tickers:
        df_d = daily_cache.get(ticker)
        if df_d is not None and not df_d.empty:
            ong_daily[ticker] = df_d
        print(".", end="", flush=True)

    nikkei_etf_df = daily_cache.get(nikkei_etf_code)
    print(f"\n  -> ONG: {len(ong_daily)}銘柄 / ETF({'OK' if nikkei_etf_df is not None else 'NG'})")

    ong_signals = generate_ong_signals(ong_daily, nikkei_etf_df, overnight_config)
    ong_signal_count = sum(
        int(df["ONG_signal"].sum()) for df in ong_signals.values()
        if "ONG_signal" in df.columns
    )
    print(f"  -> ONG シグナル総数: {ong_signal_count}")

    # ------------------------------------------------------------------
    # シンプル順張りデータ準備（5分足+日足をSM用に絞り込み）
    # ------------------------------------------------------------------
    sm_intraday: dict = {}
    sm_daily: dict = {}
    for ticker in sm_tickers:
        df_5m = intraday_plain.get(ticker)
        df_1d = daily_cache.get(ticker)
        if df_5m is not None:
            sm_intraday[ticker] = df_5m
        if df_1d is not None:
            sm_daily[ticker] = df_1d

    # ------------------------------------------------------------------
    # バックテスト実行
    # ------------------------------------------------------------------
    print("\n■ バックテスト実行...")

    print("  [午前 v12.4] 実行中...")
    morning_bt = BacktestEngine("config/strategy_config.yaml")
    morning_result = morning_bt.run(morning_signals)

    print("  [午後 リバーサル v1.2] 実行中...")
    afternoon_bt = AfternoonBacktestEngine("config/afternoon_config.yaml")
    afternoon_result = afternoon_bt.run(afternoon_signals)

    print("  [ONG v1.0] 実行中...")
    ong_bt = OvernightGapEngine("config/overnight_config.yaml")
    ong_result = ong_bt.run(ong_signals)

    print("  [シンプル順張り(値幅 ≥ 3.0%)] 実行中...")
    sm_result = sm_engine.run(sm_intraday, sm_daily)

    # ------------------------------------------------------------------
    # 合算
    # ------------------------------------------------------------------
    # 全トレード（ONG/SM は NamedTuple互換、AM/PM は engine の Trade オブジェクト）
    all_am_pm_ong_trades = morning_result.trades + afternoon_result.trades + ong_result.trades

    morning_pnl = sum(t.pnl for t in morning_result.trades)
    afternoon_pnl = sum(t.pnl for t in afternoon_result.trades)
    ong_pnl = sum(t.pnl for t in ong_result.trades)
    sm_pnl = sum(t.pnl for t in sm_result.trades)
    total_pnl = morning_pnl + afternoon_pnl + ong_pnl + sm_pnl

    combined_capital = initial_capital + ong_capital + sm_capital
    total_return = (total_pnl / combined_capital) * 100
    equity_final = combined_capital + total_pnl

    # ------------------------------------------------------------------
    # 総合エクイティカーブ（日別PnLを合算して構築）
    # ------------------------------------------------------------------
    def _collect_daily_pnl(trades, source_name: str = "") -> dict[str, float]:
        """トレードリストから日別PnLを集計する。"""
        pnl_by_date: dict[str, float] = {}
        for t in trades:
            # PairTrade (simple_momentum) は trade_date 属性を使用
            if hasattr(t, "trade_date") and t.trade_date:
                date_key = str(t.trade_date)[:10]
            elif hasattr(t, "entry_date"):
                ed = t.entry_date
                date_key = str(ed)[:10] if ed else ""
            else:
                date_key = ""
            if date_key:
                pnl_by_date[date_key] = pnl_by_date.get(date_key, 0.0) + t.pnl
        return pnl_by_date

    sm_daily_pnl = sm_result.daily_pnl  # dict[str, float]

    # AM/PM/ONG の日別PnL は trade_export.build_daily_pnl 経由で構築
    # ただし SM のトレードは PairTrade 型なので別途扱う
    all_dates: set[str] = set()
    for pnl_dict in [
        _collect_daily_pnl(morning_result.trades),
        _collect_daily_pnl(afternoon_result.trades),
        _collect_daily_pnl(ong_result.trades),
        sm_daily_pnl,
    ]:
        all_dates.update(pnl_dict.keys())

    sorted_dates = sorted(all_dates)
    combined_daily_pnl: dict[str, float] = {}
    for d in sorted_dates:
        v = (
            _collect_daily_pnl(morning_result.trades).get(d, 0.0)
            + _collect_daily_pnl(afternoon_result.trades).get(d, 0.0)
            + _collect_daily_pnl(ong_result.trades).get(d, 0.0)
            + sm_daily_pnl.get(d, 0.0)
        )
        combined_daily_pnl[d] = v

    combined_equity: list[float] = []
    running = combined_capital
    for d in sorted_dates:
        running += combined_daily_pnl[d]
        combined_equity.append(running)

    combined_max_dd = calc_dd(combined_equity)

    # ------------------------------------------------------------------
    # レポート出力
    # ------------------------------------------------------------------
    if not args.no_summary:
        margin = 3_000_000
        margin_ratio = (total_pnl / margin) * 100.0

        sm_direction = "順張り" if sm_engine.strategy.direction == "momentum" else "逆張り"
        sm_min_move = sm_engine.strategy.min_move_pct

        report = f"""
============================================================
  午前v12.4 + 午後リバーサルv1.2 + ONGv1.0
  + シンプル順張り(値幅 ≥ 3.0%)  合体レポート
============================================================

■ 総合成績
  初期資金:       {combined_capital:>14,.0f} 円
                  (AM/PM {initial_capital:,.0f} + ONG {ong_capital:,.0f} + SM {sm_capital:,.0f})
  最終資産:       {equity_final:>14,.0f} 円
  純損益:         {total_pnl:>+14,.0f} 円 ({total_return:+.2f}%)
  信用保証金:     {margin:>14,.0f} 円
  保証金比率:     {margin_ratio:>+13.2f} %  (= 純損益 / 保証金)
  最大DD(合算):   {combined_max_dd:>13.2f} %
  総トレード数:   {len(all_am_pm_ong_trades) + len(sm_result.trades)}
    午前:         {len(morning_result.trades)}件 -> {morning_pnl:>+,.0f} 円
    午後:         {len(afternoon_result.trades)}件 -> {afternoon_pnl:>+,.0f} 円
    ONG:          {len(ong_result.trades)}件 -> {ong_pnl:>+,.0f} 円
    SM:           {len(sm_result.trades)}件 -> {sm_pnl:>+,.0f} 円

■ セッション別詳細
{format_report_section("午前 v12.4 モーニング・モメンタム", morning_result.trades, initial_capital, morning_result.equity_curve)}
{format_report_section("午後 v1.2 アフタヌーン・リバーサル", afternoon_result.trades, initial_capital, afternoon_result.equity_curve)}
{format_report_section("オーバーナイト・ギャップ", ong_result.trades, ong_capital, ong_result.equity_curve)}
{format_report_section(f"シンプル順張り(値幅 ≥ {sm_min_move:.1f}%)", sm_result.trades, sm_capital, sm_result.equity_curve)}
"""
        print(report)

    # ------------------------------------------------------------------
    # Trade Export & Daily Report
    # ------------------------------------------------------------------
    # AM/PM/ONG トレードは build_trades_df が対応
    trades_df = build_trades_df(
        morning_result.trades,
        afternoon_result.trades,
        ong_result.trades,
    )

    if trades_df.empty and not sm_result.trades:
        print("\n⚠️ トレードが0件のため、日次レポート・エクスポートはスキップします。")
    else:
        last_days = args.last_days
        if not trades_df.empty:
            daily_df = build_daily_pnl(trades_df, last_days=last_days)
            print(f"\n■ 直近 {last_days} 営業日の日次損益（AM/PM/ONG）")
            print_daily_table(daily_df)

            if args.print_latest:
                latest = get_latest_day_summary(trades_df)
                print_latest_day(latest)

            if args.export:
                fmt = _detect_format(args.export)
                if fmt == "json":
                    export_trades_json(trades_df, args.export)
                else:
                    export_trades_csv(trades_df, args.export)

            if args.export_daily:
                daily_full = build_daily_pnl(trades_df, last_days=None)
                fmt = _detect_format(args.export_daily)
                if fmt == "json":
                    export_daily_json(daily_full, args.export_daily)
                else:
                    export_daily_csv(daily_full, args.export_daily)

    # ------------------------------------------------------------------
    # ファイル出力
    # ------------------------------------------------------------------
    if combined_equity:
        eq_path = os.path.join(args.output, "all_combined_equity.png")
        save_equity_curve(
            sorted_dates,
            combined_equity,
            "All Combined Strategy",
            combined_capital,
            eq_path,
        )
        print(f"\n■ 出力ファイル")
        print(f"  合算エクイティカーブ: {eq_path}")

    # SM 個別レポート出力
    if sm_result.trades:
        sm_report_text = sm_engine.generate_report(sm_result, sm_engine.initial_capital)
        sm_report_path = os.path.join(args.output, "sm_report.txt")
        with open(sm_report_path, "w", encoding="utf-8") as f:
            f.write(sm_report_text)
        logger.info(f"シンプル順張りレポート保存: {sm_report_path}")
        print(f"  SM テキストレポート:  {sm_report_path}")

        sm_eq_path = os.path.join(args.output, "sm_equity.png")
        save_equity_curve(
            sm_result.dates,
            sm_result.equity_curve,
            "Simple Momentum Strategy",
            sm_engine.initial_capital,
            sm_eq_path,
        )
        print(f"  SMエクイティカーブ:   {sm_eq_path}")

    print("\n完了\n")


if __name__ == "__main__":
    main()
