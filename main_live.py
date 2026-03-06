"""
JP Stock Auto Trading Bot v15 - BT-aligned
- Morning (9:05-11:00): EnsembleEngine (BUY + SELL) + daily bias filter
- Afternoon (config entry_start-entry_end): AfternoonReversalEngine (BUY + SELL)
- 5-min OHLCV bars from /board API
- Screener-based watchlist (same as BT)
- Short (SELL) position support: symmetric SL/TP/trailing
- Total exposure check (BT-aligned)
- Force close: PM at force_close_time, all at market close
- Cooldown: config-based bars (BT-aligned)
"""

import time
import yaml
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, date as date_type, timedelta
from core.auth import KabuAuth
from core.api_client import KabuClient
from core.order_manager import OrderManager, LivePosition
from core.bar_builder import BarBuilder
from strategy.ensemble import EnsembleEngine
from strategy.afternoon_reversal import AfternoonReversalEngine
from backtest.screener import screen_stocks

# Minimum completed 5-min bars required for signal calculation
MIN_BARS = 10

VERSION = "v15"


def load_config():
    with open("config/live_config.yaml", "r", encoding="utf-8") as f:
        live = yaml.safe_load(f)
    with open("config/strategy_config.yaml", "r", encoding="utf-8") as f:
        strategy = yaml.safe_load(f)
    try:
        with open("config/afternoon_config.yaml", "r", encoding="utf-8") as f:
            afternoon = yaml.safe_load(f)
    except FileNotFoundError:
        afternoon = None
    return live, strategy, afternoon


# ============================================
# Screener: BT-aligned watchlist
# ============================================

def run_screener(strat_cfg: dict, fallback_watchlist: list[str]) -> list[str]:
    """
    Run screen_stocks() to generate watchlist (same as BT).
    Returns list of kabuSte ticker codes (e.g. "6902").
    Falls back to config watchlist on failure.
    """
    print("\n■ Running screener (BT-aligned)...")
    try:
        selected = screen_stocks(strat_cfg)
        if not selected:
            print("  ⚠️ Screener returned 0 stocks, using fallback watchlist")
            return fallback_watchlist

        # Convert "6902.T" -> "6902" for kabuSte API
        tickers = []
        ticker_map = {}  # "6902" -> "6902.T" for yfinance
        for s in selected:
            yf_ticker = s["ticker"]  # e.g. "6902.T"
            kabu_ticker = yf_ticker.replace(".T", "")
            tickers.append(kabu_ticker)
            ticker_map[kabu_ticker] = yf_ticker

        print(f"  -> Screener selected {len(tickers)} stocks")
        for i, s in enumerate(selected[:10]):
            print(f"    {i+1}. {s['ticker']} close={s['close']:.0f} ATR%={s['atr_pct']:.2f} score={s['score']:.2f}")
        if len(selected) > 10:
            print(f"    ... and {len(selected)-10} more")

        return tickers, ticker_map
    except Exception as e:
        print(f"  ❌ Screener failed: {e}")
        print(f"  -> Using fallback watchlist ({len(fallback_watchlist)} stocks)")
        ticker_map = {t: f"{t}.T" for t in fallback_watchlist}
        return fallback_watchlist, ticker_map


# ============================================
# Daily Bias Filter (BT-aligned)
# ============================================

def calc_daily_bias(strat_cfg: dict, yf_tickers: list[str]) -> dict[str, str]:
    """
    Calculate daily bias for each ticker using yfinance daily data.
    Returns dict: yf_ticker -> "BULL" / "BEAR" / "NEUTRAL"
    BT uses daily_bias.ema_short/ema_long (default 5/25).
    """
    dc = strat_cfg.get("daily_bias", {})
    ema_short = dc.get("ema_short", 5)
    ema_long = dc.get("ema_long", 25)

    print(f"\n■ Calculating daily bias (EMA {ema_short}/{ema_long})...")
    bias = {}
    today = date_type.today()

    for yf_ticker in yf_tickers:
        try:
            data = yf.download(yf_ticker, period="60d", interval="1d", progress=False)
            if data is None or data.empty or len(data) < ema_long + 1:
                bias[yf_ticker] = "NEUTRAL"
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0].lower() for col in data.columns]
            else:
                data.columns = [col.lower() for col in data.columns]

            ema_s = ta.ema(data["close"], length=ema_short)
            ema_l = ta.ema(data["close"], length=ema_long)

            if ema_s is None or ema_l is None or pd.isna(ema_s.iloc[-1]) or pd.isna(ema_l.iloc[-1]):
                bias[yf_ticker] = "NEUTRAL"
            elif float(ema_s.iloc[-1]) > float(ema_l.iloc[-1]):
                bias[yf_ticker] = "BULL"
            else:
                bias[yf_ticker] = "BEAR"
        except Exception:
            bias[yf_ticker] = "NEUTRAL"

    bulls = sum(1 for v in bias.values() if v == "BULL")
    bears = sum(1 for v in bias.values() if v == "BEAR")
    neutrals = sum(1 for v in bias.values() if v == "NEUTRAL")
    print(f"  -> BULL={bulls} BEAR={bears} NEUTRAL={neutrals}")
    return bias


# ============================================
# Time window helpers (config-based)
# ============================================

def is_market_open() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.hour * 100 + now.minute
    return 900 <= t <= 1525


def is_morning_entry() -> bool:
    """Morning session: 9:05-11:00 (BT: _is_morning_session 900-1100)"""
    now = datetime.now()
    t = now.hour * 100 + now.minute
    return 905 <= t <= 1100


def is_afternoon_entry(entry_start: int = 1230, entry_end: int = 1400) -> bool:
    """Afternoon session: from config entry_start to entry_end"""
    now = datetime.now()
    t = now.hour * 100 + now.minute
    return entry_start <= t <= entry_end


def is_force_close_pm(force_close_time: int = 1450) -> bool:
    """PM force close time (BT: afternoon_engine force_close_time)"""
    now = datetime.now()
    t = now.hour * 100 + now.minute
    return t >= force_close_time


def is_market_close_force() -> bool:
    """Market close force liquidation (BT: daytrade引け強制決済 15:20)"""
    now = datetime.now()
    t = now.hour * 100 + now.minute
    return t >= 1520


def get_session_label(pm_start: int = 1230, pm_end: int = 1400) -> str:
    if is_morning_entry():
        return "AM"
    if is_afternoon_entry(pm_start, pm_end):
        return "PM"
    return "--"


def calc_atr_from_bars(df: pd.DataFrame, period: int = 14) -> float | None:
    """Calculate ATR from 5-min bar DataFrame."""
    if df is None or len(df) < period + 1:
        return None
    atr_series = ta.atr(df["high"], df["low"], df["close"], length=period)
    if atr_series is None or pd.isna(atr_series.iloc[-1]):
        return None
    return float(atr_series.iloc[-1])


# ============================================
# Total exposure check (BT-aligned)
# ============================================

def check_total_exposure(positions: list, new_price: float, new_size: int,
                         initial_capital: float, session: str, ticker: str,
                         now_str: str) -> int | None:
    """
    BT-aligned total exposure check.
    existing_exposure + new_notional must be <= initial_capital.
    If exceeded, auto-shrink size (Method B). Returns adjusted size or None.
    """
    existing_exposure = sum(p.entry_price * p.size for p in positions)
    new_notional = new_price * new_size
    total = existing_exposure + new_notional

    if total <= initial_capital:
        return new_size

    # How much room is left?
    remaining = initial_capital - existing_exposure
    if remaining <= 0:
        print(f"  [{now_str}] [{session}] {ticker}: SKIP (exposure full) "
              f"existing={existing_exposure:,.0f} cap={initial_capital:,.0f}")
        return None

    max_size = int(remaining / new_price)
    max_size = (max_size // 100) * 100

    if max_size < 100:
        print(f"  [{now_str}] [{session}] {ticker}: SKIP (remaining cap too small) "
              f"remaining={remaining:,.0f} price={new_price:.0f}")
        return None

    print(f"  [{now_str}] [{session}] {ticker}: size capped {new_size} -> {max_size} "
          f"(total_exposure {total:,.0f} > cap {initial_capital:,.0f})")
    return max_size


# ============================================
# Position Management: BUY/SELL symmetric
# ============================================

def update_trailing_stop(pos: LivePosition, bars_df: pd.DataFrame,
                         trail_mult: float, atr_val: float) -> None:
    if bars_df is None or len(bars_df) < 2:
        return
    current = float(bars_df["close"].iloc[-1])
    if pos.side == "BUY":
        current_high = float(bars_df["high"].iloc[-1])
        if current > pos.entry_price:
            profit_ratio = (current_high - pos.entry_price) / pos.entry_price
            accel = max(0.7, 1.0 - profit_ratio * 2.0)
            new_trail = current_high - (atr_val * trail_mult * accel)
            if new_trail > pos.trailing_stop:
                pos.trailing_stop = new_trail
    else:
        current_low = float(bars_df["low"].iloc[-1])
        if current < pos.entry_price:
            profit_ratio = (pos.entry_price - current_low) / pos.entry_price
            accel = max(0.7, 1.0 - profit_ratio * 2.0)
            new_trail = current_low + (atr_val * trail_mult * accel)
            if new_trail < pos.trailing_stop:
                pos.trailing_stop = new_trail


def check_exit(pos: LivePosition, current: float) -> str | None:
    if pos.side == "BUY":
        if current <= pos.stop_loss:
            return f"SL hit (price={current:.0f} <= SL={pos.stop_loss:.0f})"
        if current >= pos.take_profit:
            return f"TP hit (price={current:.0f} >= TP={pos.take_profit:.0f})"
        if current <= pos.trailing_stop:
            return f"Trailing hit (price={current:.0f} <= trail={pos.trailing_stop:.0f})"
    else:
        if current >= pos.stop_loss:
            return f"SL hit (price={current:.0f} >= SL={pos.stop_loss:.0f})"
        if current <= pos.take_profit:
            return f"TP hit (price={current:.0f} <= TP={pos.take_profit:.0f})"
        if current >= pos.trailing_stop:
            return f"Trailing hit (price={current:.0f} >= trail={pos.trailing_stop:.0f})"
    return None


def calc_entry_params(side: str, entry_price: float, sl_dist: float, tp_dist: float):
    if side == "BUY":
        stop_loss = entry_price - sl_dist
        take_profit = entry_price + tp_dist
        trailing_stop = stop_loss
    else:
        stop_loss = entry_price + sl_dist
        take_profit = entry_price - tp_dist
        trailing_stop = stop_loss
    return stop_loss, take_profit, trailing_stop


# ============================================
# Main
# ============================================
def main():
    live_cfg, strat_cfg, afternoon_cfg = load_config()

    # API timeout from config (fallback to 10s)
    api_timeout = live_cfg.get("api", {}).get("timeout_sec",
                  live_cfg.get("api", {}).get("timeout", 10))

    auth = KabuAuth(live_cfg["api"]["base_url"], live_cfg["api"]["password"],
                    timeout=api_timeout)
    client = KabuClient(live_cfg["api"]["base_url"], auth, timeout=api_timeout)
    order_mgr = OrderManager(client, live_cfg)

    # ================================================
    # 1) Screener: BT-aligned watchlist
    # ================================================
    use_screener = live_cfg.get("screener", {}).get("enabled", True)
    fallback_watchlist = live_cfg.get("watchlist", [])

    if use_screener:
        watchlist, ticker_map = run_screener(strat_cfg, fallback_watchlist)
    else:
        watchlist = fallback_watchlist
        ticker_map = {t: f"{t}.T" for t in watchlist}
        print(f"\n■ Screener disabled, using config watchlist ({len(watchlist)} stocks)")

    # ================================================
    # 2) PM time window from config
    # ================================================
    pm_available = afternoon_cfg is not None
    if pm_available:
        ar_cfg = afternoon_cfg["afternoon_reversal"]
        pm_entry_start = ar_cfg.get("entry_start", 1230)
        pm_entry_end = ar_cfg.get("entry_end", 1400)
        pm_force_close = afternoon_cfg["exit"].get("force_close_time", 1450)
    else:
        pm_entry_start = 1230
        pm_entry_end = 1400
        pm_force_close = 1450

    # ================================================
    # 3) Daily bias filter (AM)
    # ================================================
    yf_tickers = list(ticker_map.values())
    daily_bias = calc_daily_bias(strat_cfg, yf_tickers)

    # ================================================
    # 6) Cooldown config (BT-aligned: bars -> minutes)
    # ================================================
    am_cd = strat_cfg.get("cooldown", {})
    am_cd_enabled = am_cd.get("enabled", False)
    am_cd_loss_bars = am_cd.get("bars_after_loss", 15)
    am_cd_win_bars = am_cd.get("bars_after_win", 5)
    # Convert bars to minutes (5min bars)
    bar_interval = 5
    am_cd_loss_min = am_cd_loss_bars * bar_interval
    am_cd_win_min = am_cd_win_bars * bar_interval

    pm_cd = afternoon_cfg.get("cooldown", {}) if afternoon_cfg else {}
    pm_cd_enabled = pm_cd.get("enabled", False)
    pm_cd_loss_bars = pm_cd.get("bars_after_loss", 6)
    pm_cd_win_bars = pm_cd.get("bars_after_win", 2)
    pm_cd_loss_min = pm_cd_loss_bars * bar_interval
    pm_cd_win_min = pm_cd_win_bars * bar_interval

    # Pass cooldown config to order_manager
    order_mgr.cooldown_config = {
        "am_enabled": am_cd_enabled,
        "am_loss_min": am_cd_loss_min,
        "am_win_min": am_cd_win_min,
        "pm_enabled": pm_cd_enabled,
        "pm_loss_min": pm_cd_loss_min,
        "pm_win_min": pm_cd_win_min,
    }

    # 5-min bar builder
    bar_builder = BarBuilder(bar_interval_min=5)

    # Initialize strategy engines (backtest-compatible)
    print(f"\nLoading EnsembleEngine (AM)...")
    am_engine = EnsembleEngine("config/strategy_config.yaml")

    pm_engine = None
    if pm_available:
        print(f"Loading AfternoonReversalEngine (PM)...")
        pm_engine = AfternoonReversalEngine("config/afternoon_config.yaml")

    last_signal_time = 0
    signal_check_count = 0
    current_day = None  # for daily reset

    ensemble_cfg = strat_cfg["ensemble"]
    am_initial_capital = float(strat_cfg["global"]["initial_capital"])
    pm_initial_capital = float(afternoon_cfg["global"]["initial_capital"]) if afternoon_cfg else am_initial_capital

    print(f"\n{'='*60}")
    print(f"Bot running... ({VERSION} BT-aligned)")
    print(f"{'='*60}")
    print(f"  Watchlist: {len(watchlist)} stocks (screener={'ON' if use_screener else 'OFF'})")
    print(f"  AM Initial capital: {am_initial_capital:,.0f}")
    print(f"  PM Initial capital: {pm_initial_capital:,.0f}")
    print(f"  API timeout: {api_timeout}s")
    print(f"  [AM] Entry: 9:05-11:00 | buy_thr={ensemble_cfg['buy_threshold']} sell_thr={ensemble_cfg['sell_threshold']}")
    print(f"       SL: {strat_cfg['exit']['stop_loss_atr_multiplier']} ATR | TP: {strat_cfg['exit']['take_profit_rr_ratio']} R:R")
    print(f"       Daily bias: EMA {strat_cfg.get('daily_bias',{}).get('ema_short',5)}/{strat_cfg.get('daily_bias',{}).get('ema_long',25)} (BEAR=disabled)")
    print(f"       Cooldown: {'ON' if am_cd_enabled else 'OFF'} loss={am_cd_loss_min}min win={am_cd_win_min}min")
    if pm_available:
        print(f"  [PM] Entry: {pm_entry_start}-{pm_entry_end} | force_close={pm_force_close}")
        ar = afternoon_cfg["afternoon_reversal"]
        pm_exit = afternoon_cfg["exit"]
        print(f"       RSI<={ar['rsi_oversold']}/{ar['rsi_overbought']} + BB + VWAP + morning_move>={ar['min_morning_move_pct']}%")
        print(f"       SL: {pm_exit['stop_loss_atr_multiplier']} ATR | TP: {pm_exit['take_profit_atr_multiplier']} ATR")
        print(f"       Cooldown: {'ON' if pm_cd_enabled else 'OFF'} loss={pm_cd_loss_min}min win={pm_cd_win_min}min")
    else:
        print("  [PM] afternoon_config.yaml not found - PM session disabled")
    print(f"  Min bars: {MIN_BARS}")
    print(f"  Force close: PM@{pm_force_close} / All@1520")

    try:
        while True:
            if not is_market_open():
                time.sleep(30)
                continue

            now = datetime.now()
            now_str = now.strftime("%H:%M:%S")
            today = now.date()

            # ========================================
            # Daily reset (BT-aligned)
            # ========================================
            if current_day != today:
                if current_day is not None:
                    print(f"\n  [{now_str}] === DAY CHANGE {current_day} -> {today} ===")
                current_day = today
                order_mgr.daily_pnl = 0.0
                order_mgr.daily_trade_count = 0
                order_mgr.cooldown_until.clear()
                print(f"  [{now_str}] Daily PnL/cooldown reset")

            # ========================================
            # 4) Force close: PM positions at force_close_time, all at 1520
            # ========================================
            if is_market_close_force():
                # Close ALL remaining positions (引け強制決済)
                for pos in list(order_mgr.positions):
                    bars_df = bar_builder.get_bars(pos.ticker)
                    if not bars_df.empty:
                        current = float(bars_df["close"].iloc[-1])
                    else:
                        current = pos.entry_price
                    print(f"  [{now_str}] 引け強制決済: {pos.ticker} {pos.side}")
                    order_mgr.exit(pos, current, "引け強制決済")
            elif is_force_close_pm(pm_force_close):
                # Close PM positions only (PM force close)
                for pos in list(order_mgr.positions):
                    # Identify PM positions by entry time (after 12:00)
                    if hasattr(pos, 'entry_time') and pos.entry_time.hour >= 12:
                        bars_df = bar_builder.get_bars(pos.ticker)
                        if not bars_df.empty:
                            current = float(bars_df["close"].iloc[-1])
                        else:
                            current = pos.entry_price
                        print(f"  [{now_str}] PM時間決済({pm_force_close}): {pos.ticker} {pos.side}")
                        order_mgr.exit(pos, current, f"時間決済({pm_force_close})")

            # ========================================
            # Fetch /board and build 5-min bars
            # ========================================
            for ticker in watchlist:
                try:
                    board = client.get_board(ticker)
                    if board and board.get("CurrentPrice"):
                        bar_builder.update(ticker, board)
                except Exception as e:
                    # board timeout - continue to next ticker
                    pass
                time.sleep(0.3)

            # ========================================
            # Position management (BUY/SELL symmetric)
            # ========================================
            for pos in list(order_mgr.positions):
                bars_df = bar_builder.get_bars(pos.ticker)
                if bars_df.empty or len(bars_df) < 2:
                    continue

                current = float(bars_df["close"].iloc[-1])
                trail_mult = strat_cfg["exit"]["trailing_atr_multiplier"]
                atr_period = strat_cfg["exit"].get("atr_period", 14)

                atr_val = calc_atr_from_bars(bars_df, atr_period)
                if atr_val is None:
                    atr_val = abs(float(bars_df["close"].iloc[-1]) - float(bars_df["close"].iloc[-2]))
                    if atr_val == 0:
                        atr_val = 1.0

                update_trailing_stop(pos, bars_df, trail_mult, atr_val)

                exit_reason = check_exit(pos, current)
                if exit_reason:
                    order_mgr.exit(pos, current, exit_reason)

            # ========================================
            # Signal check
            # ========================================
            if time.time() - last_signal_time >= live_cfg["interval"]["signal_check_sec"]:
                last_signal_time = time.time()
                now = datetime.now()
                now_str = now.strftime("%H:%M:%S")
                session = get_session_label(pm_entry_start, pm_entry_end)
                signal_check_count += 1

                morning = is_morning_entry()
                afternoon = is_afternoon_entry(pm_entry_start, pm_entry_end) and pm_available

                if not morning and not afternoon:
                    if signal_check_count % 10 == 1:
                        print(f"  [{now_str}] No entry window. Positions: {len(order_mgr.positions)}")
                else:
                    if signal_check_count % 5 == 1:
                        print(f"\n  [{now_str}] === [{session}] Signal Check #{signal_check_count} ===")
                        for ticker in watchlist[:15]:  # Show first 15 for brevity
                            n_bars = bar_builder.get_bar_count(ticker)
                            bars_df = bar_builder.get_bars(ticker)
                            last_price = float(bars_df["close"].iloc[-1]) if not bars_df.empty else 0
                            ready = "OK" if n_bars >= MIN_BARS else "waiting"
                            print(f"    {ticker}: price={last_price:.0f} bars={n_bars}/{MIN_BARS} [{ready}]")
                        if len(watchlist) > 15:
                            print(f"    ... and {len(watchlist)-15} more stocks")

                    for ticker in watchlist:
                        if not order_mgr.can_entry(ticker):
                            continue

                        bars_df = bar_builder.get_bars(ticker)
                        n_bars = bar_builder.get_bar_count(ticker)

                        if n_bars < MIN_BARS:
                            continue

                        # ========== MORNING: EnsembleEngine ==========
                        if morning:
                            # 3) Daily bias filter: BEAR -> skip AM
                            yf_ticker = ticker_map.get(ticker, f"{ticker}.T")
                            bias = daily_bias.get(yf_ticker, "NEUTRAL")
                            if bias == "BEAR":
                                # BT: apply_v11_filter sets HOLD on BEAR days
                                continue

                            signal, score, detail = am_engine.evaluate_live(bars_df)

                            if score != 0:
                                print(f"  [{now_str}] [AM] {ticker}: score={score:.2f} signal={signal} bias={bias} ({detail})")

                            if signal in ("BUY", "SELL"):
                                current = float(bars_df["close"].iloc[-1])

                                # VWAP filter
                                vwap_val = None
                                if "vwap" in bars_df.columns:
                                    vwap_val = float(bars_df["vwap"].iloc[-1]) if not pd.isna(bars_df["vwap"].iloc[-1]) else None

                                if vwap_val is not None:
                                    if signal == "BUY" and current < vwap_val:
                                        print(f"  [{now_str}] [AM] {ticker}: BUY blocked by VWAP (price={current:.0f} < vwap={vwap_val:.0f})")
                                        continue
                                    if signal == "SELL" and current > vwap_val:
                                        print(f"  [{now_str}] [AM] {ticker}: SELL blocked by VWAP (price={current:.0f} > vwap={vwap_val:.0f})")
                                        continue

                                atr_val = calc_atr_from_bars(bars_df, strat_cfg["exit"].get("atr_period", 14))
                                if atr_val is None:
                                    atr_val = max(current * 0.01, 1)

                                sl_dist = atr_val * strat_cfg["exit"]["stop_loss_atr_multiplier"]
                                sl_dist = max(sl_dist, 1)
                                tp_dist = sl_dist * strat_cfg["exit"]["take_profit_rr_ratio"]

                                risk_per = strat_cfg["global"]["risk_per_trade"]
                                size = int((am_initial_capital * risk_per) / sl_dist)
                                size = max((size // 100) * 100, 100)

                                # 5) Total exposure check (BT-aligned)
                                size = check_total_exposure(
                                    order_mgr.positions, current, size,
                                    am_initial_capital, "AM", ticker, now_str)
                                if size is None:
                                    continue

                                entry_price = current
                                stop_loss, take_profit, trailing_stop = calc_entry_params(
                                    signal, entry_price, sl_dist, tp_dist)

                                # BT expected price for comparison
                                slip = strat_cfg["global"].get("slippage_rate", 0.001)
                                if signal == "BUY":
                                    bt_expected = entry_price * (1 + slip)
                                else:
                                    bt_expected = entry_price * (1 - slip)

                                reason = f"AM Ensemble score={score:.2f} bias={bias} ({detail})"
                                print(f"  [{now_str}] [AM] >>> ENTRY {signal} {ticker}: price={entry_price:.0f} "
                                      f"(BT_est={bt_expected:.0f}) SL={stop_loss:.0f} TP={take_profit:.0f} size={size}")
                                order_mgr.entry(ticker, signal, entry_price, size,
                                                stop_loss, take_profit, reason,
                                                session="AM")

                        # ========== AFTERNOON: AfternoonReversalEngine ==========
                        if afternoon:
                            signal, info = pm_engine.evaluate_live(bars_df)
                            rsi_val = info.get("rsi")
                            mm_val = info.get("morning_move")

                            if rsi_val is not None and (rsi_val < 35 or rsi_val > 65):
                                print(f"  [{now_str}] [PM] {ticker}: RSI={rsi_val:.1f} morning_move={mm_val:.1f}% signal={signal}")

                            if signal in ("BUY", "SELL"):
                                current = float(bars_df["close"].iloc[-1])

                                pm_exit = afternoon_cfg["exit"]
                                atr_val = calc_atr_from_bars(bars_df, pm_exit.get("atr_period", 14))
                                if atr_val is None:
                                    atr_val = max(current * 0.01, 1)

                                sl_dist = atr_val * pm_exit["stop_loss_atr_multiplier"]
                                sl_dist = max(sl_dist, 1)
                                tp_dist = atr_val * pm_exit["take_profit_atr_multiplier"]

                                pm_global = afternoon_cfg["global"]
                                risk_per = pm_global["risk_per_trade"]
                                size = int((pm_initial_capital * risk_per) / sl_dist)
                                size = max((size // 100) * 100, 100)

                                # 5) Total exposure check (BT-aligned)
                                size = check_total_exposure(
                                    order_mgr.positions, current, size,
                                    pm_initial_capital, "PM", ticker, now_str)
                                if size is None:
                                    continue

                                entry_price = current
                                stop_loss, take_profit, trailing_stop = calc_entry_params(
                                    signal, entry_price, sl_dist, tp_dist)

                                slip = pm_global.get("slippage_rate", 0.001)
                                if signal == "BUY":
                                    bt_expected = entry_price * (1 + slip)
                                else:
                                    bt_expected = entry_price * (1 - slip)

                                reason = f"PM Reversal RSI={rsi_val:.1f} mm={mm_val:.1f}%"
                                print(f"  [{now_str}] [PM] >>> ENTRY {signal} {ticker}: price={entry_price:.0f} "
                                      f"(BT_est={bt_expected:.0f}) SL={stop_loss:.0f} TP={take_profit:.0f} size={size} RSI={rsi_val:.1f}")
                                order_mgr.entry(ticker, signal, entry_price, size,
                                                stop_loss, take_profit, reason,
                                                session="PM")

            time.sleep(live_cfg["interval"]["price_check_sec"])

    except KeyboardInterrupt:
        print("\nStopped.")
        print(f"  Total signal checks: {signal_check_count}")
        print(f"  Open positions: {len(order_mgr.positions)}")
        if order_mgr.trades:
            print(order_mgr.get_daily_summary())


if __name__ == "__main__":
    main()
