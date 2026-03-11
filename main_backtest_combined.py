"""
合体バックテスト: 午前 v12.4 + 午後リバーサル v1.2
- 午前（9:00-12:00）: モーニング・モメンタム順張り
- 午後（12:30-14:00）: アフタヌーン・リバーサル逆張り

CLI Options:
  --last-days N        直近N営業日の日次レポート (default: 7)
  --export FILE        トレード明細をCSV/JSON出力 (拡張子で判定)
  --export-daily FILE  日次集計をCSV/JSON出力
  --print-latest       最新日のAM/PM別損益をコンソール表示
  --no-summary         総合サマリ出力を抑制
"""

import argparse
import pandas as pd
import yfinance as yf
import yaml
import pandas_ta as ta
from backtest.engine import BacktestEngine
from backtest.afternoon_engine import AfternoonBacktestEngine
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


def _is_morning_session(timestamp):
    """午前v12.4: 12:00までのシグナルのみ有効にするフィルター"""
    if not hasattr(timestamp, "hour"):
        return True
    t = timestamp.hour * 100 + timestamp.minute
    return 900 <= t <= 1200


def apply_v11_filter(signals_df, daily_bias):
    result = signals_df.copy()
    for idx in result.index:
        date = idx.date() if hasattr(idx, "date") else idx
        
        # 1. 日足バイアスフィルター
        b = daily_bias.get(date, "NEUTRAL")
        if b == "BEAR":
            result.loc[idx, "final_signal"] = "HOLD"
            
        # 2. 時間帯フィルター（12:00以降は新規エントリーしない）
        if not _is_morning_session(idx):
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


def _detect_format(filepath: str) -> str:
    """Detect file format from extension."""
    lower = filepath.lower()
    if lower.endswith(".json"):
        return "json"
    return "csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="合体バックテスト: 午前 v12.4 + 午後リバーサル v1.2"
    )
    parser.add_argument(
        "--last-days", type=int, default=7,
        help="直近N営業日の日次レポートを表示 (default: 7)"
    )
    parser.add_argument(
        "--export", type=str, default=None, metavar="FILE",
        help="トレード明細をCSV/JSONで出力 (拡張子で判定)"
    )
    parser.add_argument(
        "--export-daily", type=str, default=None, metavar="FILE",
        help="日次集計をCSV/JSONで出力"
    )
    parser.add_argument(
        "--print-latest", action="store_true",
        help="最新日（データ上の最終日）のAM/PM別損益を表示"
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="総合サマリ出力を抑制"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  午前v12.4 + 午後リバーサルv1.2 合体バックテスト")
    print("=" * 60)

    # === 設定読み込み ===
    with open("config/strategy_config.yaml", "r", encoding="utf-8") as f:
        morning_config = yaml.safe_load(f)
    with open("config/afternoon_config.yaml", "r", encoding="utf-8") as f:
        afternoon_config = yaml.safe_load(f)

    initial_capital = morning_config["global"]["initial_capital"]

    # combined_max_positions: AM/PMで共有するポジション上限（本番との整合）
    combined_max_positions = morning_config["global"].get("combined_max_positions", None)

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
    if combined_max_positions is not None:
        morning_engine.max_positions = combined_max_positions
        print(f"  -> combined_max_positions={combined_max_positions} を適用（AM）")
    morning_result = morning_engine.run(morning_signals)

    # AM終了後、PM開始時点（12:30）でまだオープンしているAMポジション数を集計
    # → PM側がスロットを適切に空けるための情報として渡す
    def _count_am_positions_held_into_pm(morning_trades, pm_start=1230):
        """AMポジションのうちPMセッション開始時刻（12:30）以降も保有していたものを日付別に集計"""
        from collections import defaultdict
        counts = defaultdict(int)
        for trade in morning_trades:
            exit_t = trade.exit_date
            if not hasattr(exit_t, "hour"):
                continue
            exit_time_int = exit_t.hour * 100 + exit_t.minute
            if exit_time_int >= pm_start:
                day = exit_t.date() if hasattr(exit_t, "date") else exit_t
                counts[day] += 1
        return dict(counts)

    am_open_at_pm = _count_am_positions_held_into_pm(morning_result.trades)

    # 午後
    print("  [午後 リバーサル v1.2] 実行中...")
    afternoon_engine = AfternoonBacktestEngine("config/afternoon_config.yaml")
    if combined_max_positions is not None:
        afternoon_engine.max_positions = combined_max_positions
        print(f"  -> combined_max_positions={combined_max_positions} を適用（PM）")
    afternoon_result = afternoon_engine.run(afternoon_signals, am_open_per_day=am_open_at_pm)

    # === 合算 ===
    all_trades = morning_result.trades + afternoon_result.trades
    total_pnl = sum(t.pnl for t in all_trades)
    total_return = (total_pnl / initial_capital) * 100
    equity_final = initial_capital + total_pnl

    morning_pnl = sum(t.pnl for t in morning_result.trades)
    afternoon_pnl = sum(t.pnl for t in afternoon_result.trades)

    # 合算DD（簡易: 各セッションの最大DDの大きい方）
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

    # === レポート出力（従来の総合サマリ）===
    if not args.no_summary:
        report = f"""
============================================================
  午前v12.4 + 午後リバーサルv1.2 合体レポート
============================================================

■ 総合成績
  初期資金:       {initial_capital:>14,.0f} 円
  最終資産:       {equity_final:>14,.0f} 円
  純損益:         {total_pnl:>+14,.0f} 円 ({total_return:+.2f}%)
  総トレード数:   {len(all_trades)}
    午前:         {len(morning_result.trades)}件 -> {morning_pnl:>+,.0f} 円
    午後:         {len(afternoon_result.trades)}件 -> {afternoon_pnl:>+,.0f} 円

■ セッション別詳細
{format_report_section("午前 v12.4 モーニング・モメンタム", morning_result.trades, initial_capital, morning_result.equity_curve)}
{format_report_section("午後 v1.2 アフタヌーン・リバーサル", afternoon_result.trades, initial_capital, afternoon_result.equity_curve)}
"""
        print(report)

    # ===========================================================
    # Trade Export & Daily Report (new features)
    # ===========================================================

    # Build unified trades DataFrame
    trades_df = build_trades_df(morning_result.trades, afternoon_result.trades)

    if trades_df.empty:
        print("\n⚠️ トレードが0件のため、日次レポート・エクスポートはスキップします。")
    else:
        # --- Daily PnL table (always show last N days) ---
        last_days = args.last_days
        daily_df = build_daily_pnl(trades_df, last_days=last_days)

        print(f"\n■ 直近 {last_days} 営業日の日次損益")
        print_daily_table(daily_df)

        # --- Latest day summary ---
        if args.print_latest:
            latest = get_latest_day_summary(trades_df)
            print_latest_day(latest)

        # --- Export trade detail ---
        if args.export:
            fmt = _detect_format(args.export)
            if fmt == "json":
                export_trades_json(trades_df, args.export)
            else:
                export_trades_csv(trades_df, args.export)

        # --- Export daily PnL ---
        if args.export_daily:
            # Export full daily (not limited to last_days)
            daily_full = build_daily_pnl(trades_df, last_days=None)
            fmt = _detect_format(args.export_daily)
            if fmt == "json":
                export_daily_json(daily_full, args.export_daily)
            else:
                export_daily_csv(daily_full, args.export_daily)

    print("\n完了")


if __name__ == "__main__":
    main()
