"""
JP Stock Auto Trading Bot v13
- Morning (9:05-11:00): EnsembleEngine (BUY + SELL)
- Afternoon (12:35-14:00): AfternoonReversalEngine (BUY + SELL)
- 5-min OHLCV bars from /board API
- Short (SELL) position support: symmetric SL/TP/trailing
- Notional cap: auto-shrink size when order value > initial_capital
"""

import time
import yaml
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from core.auth import KabuAuth
from core.api_client import KabuClient
from core.order_manager import OrderManager, LivePosition
from core.bar_builder import BarBuilder
from strategy.ensemble import EnsembleEngine
from strategy.afternoon_reversal import AfternoonReversalEngine

# Minimum completed 5-min bars required for signal calculation
MIN_BARS = 10

VERSION = "v13"


def load_config():
    with open("config/live_config.yaml", "r", encoding="utf-8") as f:
        live = yaml.safe_load(f)
    with open("config/strategy_config.yaml", "r", encoding="utf-8") as f:
        strategy = yaml.safe_load(f)
    # Load afternoon config
    try:
        with open("config/afternoon_config.yaml", "r", encoding="utf-8") as f:
            afternoon = yaml.safe_load(f)
    except FileNotFoundError:
        afternoon = None
    return live, strategy, afternoon


def is_market_open() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.hour * 100 + now.minute
    return 900 <= t <= 1525


def is_morning_entry() -> bool:
    """Morning session: 9:05-11:00"""
    now = datetime.now()
    t = now.hour * 100 + now.minute
    return 905 <= t <= 1100


def is_afternoon_entry() -> bool:
    """Afternoon session: 12:35-14:00"""
    now = datetime.now()
    t = now.hour * 100 + now.minute
    return 1235 <= t <= 1400


def get_session_label() -> str:
    if is_morning_entry():
        return "AM"
    if is_afternoon_entry():
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
# Notional cap helper (Method B: auto-shrink)
# ============================================

def cap_size_by_notional(size: int, entry_price: float, max_capital: float,
                         session: str, ticker: str, now_str: str) -> int | None:
    """
    Check if notional (entry_price * size) exceeds max_capital.
    If it does, shrink size to fit within max_capital (100-share units).
    Returns adjusted size, or None if even 100 shares exceed cap.
    """
    notional = entry_price * size
    if notional <= max_capital:
        return size  # no adjustment needed

    max_size = int(max_capital / entry_price)
    max_size = (max_size // 100) * 100  # round down to 100-share unit

    if max_size < 100:
        print(f"  [{now_str}] [{session}] {ticker}: SKIP (cap too small) "
              f"cap={max_capital:,.0f} price={entry_price:.0f}")
        return None

    print(f"  [{now_str}] [{session}] {ticker}: size capped {size} -> {max_size} "
          f"(notional {notional:,.0f} > cap {max_capital:,.0f})")
    return max_size


# ============================================
# Position Management: BUY/SELL symmetric
# ============================================

def update_trailing_stop(pos: LivePosition, bars_df: pd.DataFrame,
                         trail_mult: float, atr_val: float) -> None:
    """
    Update trailing stop for a position.
    BUY:  trail up  using high  -> trailing_stop rises
    SELL: trail down using low  -> trailing_stop falls
    Both use accelerated trailing (v12.4 style).
    """
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
        # SELL (short): trailing stop is ABOVE current price
        current_low = float(bars_df["low"].iloc[-1])
        if current < pos.entry_price:
            profit_ratio = (pos.entry_price - current_low) / pos.entry_price
            accel = max(0.7, 1.0 - profit_ratio * 2.0)
            new_trail = current_low + (atr_val * trail_mult * accel)
            if new_trail < pos.trailing_stop:
                pos.trailing_stop = new_trail


def check_exit(pos: LivePosition, current: float) -> str | None:
    """
    Check exit conditions for BUY or SELL position.
    Returns exit reason string, or None if no exit.
    """
    if pos.side == "BUY":
        if current <= pos.stop_loss:
            return f"SL hit (price={current:.0f} <= SL={pos.stop_loss:.0f})"
        if current >= pos.take_profit:
            return f"TP hit (price={current:.0f} >= TP={pos.take_profit:.0f})"
        if current <= pos.trailing_stop:
            return f"Trailing hit (price={current:.0f} <= trail={pos.trailing_stop:.0f})"
    else:
        # SELL (short): exits are mirrored
        if current >= pos.stop_loss:
            return f"SL hit (price={current:.0f} >= SL={pos.stop_loss:.0f})"
        if current <= pos.take_profit:
            return f"TP hit (price={current:.0f} <= TP={pos.take_profit:.0f})"
        if current >= pos.trailing_stop:
            return f"Trailing hit (price={current:.0f} >= trail={pos.trailing_stop:.0f})"
    return None


# ============================================
# Entry helpers
# ============================================

def calc_entry_params(side: str, entry_price: float, sl_dist: float, tp_dist: float):
    """
    Calculate SL, TP, initial trailing_stop for BUY or SELL.
    Returns (stop_loss, take_profit, trailing_stop)
    """
    if side == "BUY":
        stop_loss = entry_price - sl_dist
        take_profit = entry_price + tp_dist
        trailing_stop = stop_loss  # initial trailing = SL
    else:
        stop_loss = entry_price + sl_dist
        take_profit = entry_price - tp_dist
        trailing_stop = stop_loss  # initial trailing = SL (above entry)
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

    # 5-min bar builder
    bar_builder = BarBuilder(bar_interval_min=5)

    # Initialize strategy engines (backtest-compatible)
    print(f"Loading EnsembleEngine (AM)...")
    am_engine = EnsembleEngine("config/strategy_config.yaml")

    pm_engine = None
    pm_available = afternoon_cfg is not None
    if pm_available:
        print(f"Loading AfternoonReversalEngine (PM)...")
        pm_engine = AfternoonReversalEngine("config/afternoon_config.yaml")

    last_signal_time = 0
    signal_check_count = 0

    ensemble_cfg = strat_cfg["ensemble"]

    print(f"\nBot running... ({VERSION} AM Ensemble + PM Reversal)")
    print(f"  Watchlist: {live_cfg['watchlist']}")
    print(f"  Initial capital: {live_cfg['trade']['initial_capital']:,}")
    print(f"  API timeout: {api_timeout}s")
    print(f"  [AM] Entry: 9:05-11:00 | buy_thr={ensemble_cfg['buy_threshold']} sell_thr={ensemble_cfg['sell_threshold']}")
    print(f"       SL: {strat_cfg['exit']['stop_loss_atr_multiplier']} ATR | TP: {strat_cfg['exit']['take_profit_rr_ratio']} R:R")
    if pm_available:
        ar = afternoon_cfg["afternoon_reversal"]
        pm_exit = afternoon_cfg["exit"]
        print(f"  [PM] Entry: 12:35-14:00 | RSI<={ar['rsi_oversold']}/{ar['rsi_overbought']} + BB + VWAP + morning_move>={ar['min_morning_move_pct']}%")
        print(f"       SL: {pm_exit['stop_loss_atr_multiplier']} ATR | TP: {pm_exit['take_profit_atr_multiplier']} ATR")
    else:
        print("  [PM] afternoon_config.yaml not found - PM session disabled")
    print(f"  Min bars: {MIN_BARS}")

    try:
        while True:
            if not is_market_open():
                time.sleep(30)
                continue

            # ========================================
            # Fetch /board and build 5-min bars
            # ========================================
            for ticker in live_cfg["watchlist"]:
                board = client.get_board(ticker)
                if board and board.get("CurrentPrice"):
                    bar_builder.update(ticker, board)
                time.sleep(0.5)

            # ========================================
            # Position management (BUY/SELL symmetric)
            # ========================================
            for pos in list(order_mgr.positions):
                bars_df = bar_builder.get_bars(pos.ticker)
                if bars_df.empty or len(bars_df) < 2:
                    continue

                current = float(bars_df["close"].iloc[-1])

                # Determine which config to use for trailing
                # (Use AM exit config as default; PM positions use PM config)
                trail_mult = strat_cfg["exit"]["trailing_atr_multiplier"]
                atr_period = strat_cfg["exit"].get("atr_period", 14)

                atr_val = calc_atr_from_bars(bars_df, atr_period)
                if atr_val is None:
                    atr_val = abs(float(bars_df["close"].iloc[-1]) - float(bars_df["close"].iloc[-2]))
                    if atr_val == 0:
                        atr_val = 1.0

                # Update trailing stop (BUY/SELL symmetric)
                update_trailing_stop(pos, bars_df, trail_mult, atr_val)

                # Exit check (BUY/SELL symmetric)
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
                session = get_session_label()
                signal_check_count += 1

                morning = is_morning_entry()
                afternoon = is_afternoon_entry() and pm_available

                if not morning and not afternoon:
                    if signal_check_count % 10 == 1:
                        print(f"  [{now_str}] No entry window. Positions: {len(order_mgr.positions)}")
                else:
                    # Log data status every 5 checks
                    if signal_check_count % 5 == 1:
                        print(f"\n  [{now_str}] === [{session}] Signal Check #{signal_check_count} ===")
                        for ticker in live_cfg["watchlist"]:
                            n_bars = bar_builder.get_bar_count(ticker)
                            bars_df = bar_builder.get_bars(ticker)
                            last_price = float(bars_df["close"].iloc[-1]) if not bars_df.empty else 0
                            ready = "OK" if n_bars >= MIN_BARS else "waiting"
                            print(f"    {ticker}: price={last_price:.0f} bars={n_bars}/{MIN_BARS} [{ready}]")

                    for ticker in live_cfg["watchlist"]:
                        if not order_mgr.can_entry(ticker):
                            continue

                        bars_df = bar_builder.get_bars(ticker)
                        n_bars = bar_builder.get_bar_count(ticker)

                        if n_bars < MIN_BARS:
                            continue

                        # ========== MORNING: EnsembleEngine ==========
                        if morning:
                            signal, score, detail = am_engine.evaluate_live(bars_df)

                            if score != 0:
                                print(f"  [{now_str}] [AM] {ticker}: score={score:.2f} signal={signal} ({detail})")

                            if signal in ("BUY", "SELL"):
                                current = float(bars_df["close"].iloc[-1])

                                # VWAP filter (BUY: price > VWAP, SELL: price < VWAP)
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

                                risk_per = live_cfg["trade"]["risk_per_trade"]
                                size = int((live_cfg["trade"]["initial_capital"] * risk_per) / sl_dist)
                                size = max((size // 100) * 100, 100)

                                # --- Notional cap (Method B: auto-shrink) ---
                                max_cap = float(live_cfg["trade"]["initial_capital"])
                                size = cap_size_by_notional(
                                    size, current, max_cap, "AM", ticker, now_str)
                                if size is None:
                                    continue
                                # --- End notional cap ---

                                entry_price = current
                                stop_loss, take_profit, trailing_stop = calc_entry_params(
                                    signal, entry_price, sl_dist, tp_dist)

                                reason = f"AM Ensemble score={score:.2f} ({detail})"
                                print(f"  [{now_str}] [AM] >>> ENTRY {signal} {ticker}: price={entry_price:.0f} SL={stop_loss:.0f} TP={take_profit:.0f} size={size}")
                                order_mgr.entry(ticker, signal, entry_price, size, stop_loss, take_profit, reason)

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
                                size = int((pm_global["initial_capital"] * risk_per) / sl_dist)
                                size = max((size // 100) * 100, 100)

                                # --- Notional cap (Method B: auto-shrink) ---
                                max_cap = float(pm_global["initial_capital"])
                                size = cap_size_by_notional(
                                    size, current, max_cap, "PM", ticker, now_str)
                                if size is None:
                                    continue
                                # --- End notional cap ---

                                entry_price = current
                                stop_loss, take_profit, trailing_stop = calc_entry_params(
                                    signal, entry_price, sl_dist, tp_dist)

                                reason = f"PM Reversal RSI={rsi_val:.1f} mm={mm_val:.1f}%"
                                print(f"  [{now_str}] [PM] >>> ENTRY {signal} {ticker}: price={entry_price:.0f} SL={stop_loss:.0f} TP={take_profit:.0f} size={size} RSI={rsi_val:.1f}")
                                order_mgr.entry(ticker, signal, entry_price, size, stop_loss, take_profit, reason)

            time.sleep(live_cfg["interval"]["price_check_sec"])

    except KeyboardInterrupt:
        print("\nStopped.")
        print(f"  Total signal checks: {signal_check_count}")
        print(f"  Open positions: {len(order_mgr.positions)}")
        if order_mgr.trades:
            print(order_mgr.get_daily_summary())


if __name__ == "__main__":
    main()
