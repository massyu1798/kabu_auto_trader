"""
統合バックテスト v21: 午前 v12.4 + 午後リバーサル v1.2 + オーバーナイト・ギャップ
- 午前（9:00-12:00）: モーニング・モメンタム順張り
- 午後（12:30-14:00）: アフタヌーン・リバーサル逆張り
- 夜間（引け買い→翌寄り売り）: オーバーナイト・ギャップ

CLI Options:
  --last-days N        直近N営業日の日次レポート (default: 7)
  --export FILE        トレード明細をCSV/JSON出力 (拡張子で判定)
  --export-daily FILE  日次集計をCSV/JSON出力
  --print-latest       最新日のAM/PM/ONG別損益をコンソール表示
  --no-summary         総合サマリ出力を抑制
"""

import argparse
import pandas as pd
import yfinance as yf
import yaml
import pandas_ta as ta
from backtest.engine import BacktestEngine
from backtest.afternoon_engine import AfternoonBacktestEngine
from backtest.overnight_engine import OvernightGapEngine
from backtest.screener import screen_stocks, STOCK_POOL
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
from risk.risk_manager import RiskManager


# ユニバース: ONG専用（日経225 ETF, TOPIX ETF + 既存STOCK_POOLから上位20銘柄）
ONG_ETF_TICKERS = ["1321.T", "1306.T"]
ONG_STOCK_TICKERS = STOCK_POOL[:20]  # 時価総額上位20銘柄相当
ONG_UNIVERSE = ONG_ETF_TICKERS + ONG_STOCK_TICKERS


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


def load_daily(ticker, period="120d"):
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0].lower() for col in data.columns]
        else:
            data.columns = [col.lower() for col in data.columns]
        data = data[["open", "high", "low", "close", "volume"]].copy()
        data.dropna(inplace=True)
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


def calc_dd(equity_curve):
    """最大ドローダウン（%）を計算"""
    if not equity_curve:
        return 0
    peak = equity_curve[0]
    max_dd = 0
    for e in equity_curve:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def format_strategy_section(title, trades, initial_capital, equity_curve):
    """戦略別成績セクションをフォーマット"""
    if not trades:
        return f"\n  {title}: トレードなし\n"

    total_trades = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

    total_pnl = sum(t.pnl for t in trades)
    total_return = (total_pnl / initial_capital) * 100

    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

    total_win_amt = sum(t.pnl for t in wins)
    total_loss_amt = abs(sum(t.pnl for t in losses))
    pf = (total_win_amt / total_loss_amt) if total_loss_amt != 0 else float("inf")

    max_dd = calc_dd(equity_curve)

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
        description="統合バックテスト v21: 午前 + 午後 + オーバーナイト・ギャップ"
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
        help="最新日（データ上の最終日）のAM/PM/ONG別損益を表示"
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="総合サマリ出力を抑制"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  統合バックテストレポート v21")
    print("=" * 60)

    # === 設定読み込み ===
    with open("config/strategy_config.yaml", "r", encoding="utf-8") as f:
        morning_config = yaml.safe_load(f)
    with open("config/afternoon_config.yaml", "r", encoding="utf-8") as f:
        afternoon_config = yaml.safe_load(f)

    initial_capital = morning_config["global"]["initial_capital"]
    combined_max_positions = morning_config["global"].get("combined_max_positions", None)

    # === 銘柄スクリーニング（AM/PM用）===
    print("\n■ 銘柄スクリーニング（AM/PM）...")
    selected = screen_stocks(morning_config)
    tickers = [s["ticker"] for s in selected]
    print(f"  -> {len(tickers)}銘柄を選定")

    # === データ取得 + シグナル生成（AM/PM）===
    print("\n■ AM/PM解析中...")

    ensemble = EnsembleEngine("config/strategy_config.yaml")
    morning_signals = {}

    reversal_engine = AfternoonReversalEngine("config/afternoon_config.yaml")
    afternoon_signals = {}

    for ticker in tickers:
        df_5m = load_intraday(ticker)
        df_daily = load_daily(ticker)
        if df_5m is None or len(df_5m) < 20:
            continue

        if df_daily is not None:
            bias = calc_daily_bias(df_daily, morning_config)
            signals_df = ensemble.generate_ensemble_signals(df_5m)
            signals_df = apply_v11_filter(signals_df, bias)
            morning_signals[ticker] = signals_df

        afternoon_df = reversal_engine.generate_signals(df_5m)
        afternoon_signals[ticker] = afternoon_df

        print(".", end="", flush=True)

    print(f"\n  -> 午前: {len(morning_signals)}銘柄 / 午後: {len(afternoon_signals)}銘柄")

    # === ONG データ取得 ===
    print("\n■ ONG（オーバーナイト・ギャップ）データ取得中...")
    ong_daily_data = {}
    nikkei_daily = None

    for ticker in ONG_UNIVERSE:
        df_d = load_daily(ticker, period="120d")
        if df_d is not None and len(df_d) >= 10:
            ong_daily_data[ticker] = df_d
            print(".", end="", flush=True)

    # 日経225 ETF(1321.T): 条件3（夜間追い風近似）用
    if "1321.T" in ong_daily_data:
        nikkei_daily = ong_daily_data["1321.T"]

    print(f"\n  -> ONG: {len(ong_daily_data)}銘柄")

    # === AM/PM バックテスト ===
    print("\n■ バックテスト実行...")

    print("  [午前 v12.4] 実行中...")
    morning_engine = BacktestEngine("config/strategy_config.yaml")
    if combined_max_positions is not None:
        morning_engine.max_positions = combined_max_positions
    morning_result = morning_engine.run(morning_signals)

    def _count_am_positions_held_into_pm(morning_trades, pm_start=1230):
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

    print("  [午後 リバーサル v1.2] 実行中...")
    afternoon_engine = AfternoonBacktestEngine("config/afternoon_config.yaml")
    if combined_max_positions is not None:
        afternoon_engine.max_positions = combined_max_positions
    afternoon_result = afternoon_engine.run(afternoon_signals, am_open_per_day=am_open_at_pm)

    # === ONG バックテスト ===
    print("  [オーバーナイト・ギャップ] 実行中...")
    ong_engine = OvernightGapEngine("config/strategy_config.yaml")
    if nikkei_daily is not None:
        ong_engine.set_nikkei_daily(nikkei_daily)
    ong_result = ong_engine.run(ong_daily_data)

    # === リスク管理サマリ（3戦略合算） ===
    risk_manager = RiskManager(initial_capital=initial_capital)
    # 全トレードを時系列順に処理してリスクメトリクスを集計
    all_trades_sorted = sorted(
        morning_result.trades + afternoon_result.trades + ong_result.trades,
        key=lambda t: t.entry_date,
    )
    for t in all_trades_sorted:
        risk_manager.update_day(t.entry_date)
        risk_manager.record_trade_pnl(t.pnl)

    # === 合算集計 ===
    all_trades = morning_result.trades + afternoon_result.trades + ong_result.trades
    total_pnl = sum(t.pnl for t in all_trades)
    total_return = (total_pnl / initial_capital) * 100
    equity_final = initial_capital + total_pnl

    morning_pnl = sum(t.pnl for t in morning_result.trades)
    afternoon_pnl = sum(t.pnl for t in afternoon_result.trades)
    ong_pnl = sum(t.pnl for t in ong_result.trades)

    # === レポート出力 ===
    if not args.no_summary:
        report = f"""
============================================================
  統合バックテストレポート v21
============================================================

■ 総合成績（全戦略合算）
  初期資金:       {initial_capital:>14,.0f} 円
  最終資産:       {equity_final:>14,.0f} 円
  純損益:         {total_pnl:>+14,.0f} 円 ({total_return:+.2f}%)
  総トレード数:   {len(all_trades)}
    午前:         {len(morning_result.trades)}件 -> {morning_pnl:>+,.0f} 円
    午後:         {len(afternoon_result.trades)}件 -> {afternoon_pnl:>+,.0f} 円
    ONG:          {len(ong_result.trades)}件 -> {ong_pnl:>+,.0f} 円

■ 戦略別成績
{format_strategy_section("[1] 午前 v12.4 モーニング・モメンタム（MR+BO）", morning_result.trades, initial_capital, morning_result.equity_curve)}
{format_strategy_section("[2] 午後 v1.2 アフタヌーン・リバーサル", afternoon_result.trades, initial_capital, afternoon_result.equity_curve)}
{format_strategy_section("[3] オーバーナイト・ギャップ", ong_result.trades, initial_capital, ong_result.equity_curve)}
"""
        print(report)

        # リスク管理レポート
        print(risk_manager.format_report())
        print(f"  最大同時ポジション数:    {ong_result.max_concurrent_positions:>4} 件（ONG）")
        print(f"  ONG日次停止回数:         {ong_result.daily_halt_count:>4} 回")
        print(f"  ONG週間停止回数:         {ong_result.weekly_halt_count:>4} 回")
        print(f"  ONG月間停止回数:         {ong_result.monthly_halt_count:>4} 回")

    # ===========================================================
    # Trade Export & Daily Report
    # ===========================================================
    trades_df = build_trades_df(
        morning_result.trades,
        afternoon_result.trades,
        ong_result.trades,
    )

    if trades_df.empty:
        print("\n⚠️ トレードが0件のため、日次レポート・エクスポートはスキップします。")
    else:
        last_days = args.last_days
        daily_df = build_daily_pnl(trades_df, last_days=last_days)

        print(f"\n■ 直近 {last_days} 営業日の日次損益（AM / PM / ONG別）")
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

    print("\n完了")


if __name__ == "__main__":
    main()
