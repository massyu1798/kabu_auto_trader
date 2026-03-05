"""
JP Stock Auto Trading Bot v12.4 + Afternoon Reversal v1.2
- Morning (9:05-11:00): Momentum + VWAP filter
- Afternoon (12:35-14:00): RSI + BB reversal
"""

import time
import yaml
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from core.auth import KabuAuth
from core.api_client import KabuClient
from core.order_manager import OrderManager, LivePosition

# Minimum data points for signal calculation
MIN_DATA_POINTS = 30

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
    if now.weekday() >= 5: return False
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
    if is_morning_entry(): return "AM"
    if is_afternoon_entry(): return "PM"
    return "--"

def calc_atr(price_hist: dict, period: int = 14) -> float | None:
    """Calculate ATR using pandas_ta"""
    h = price_hist["high"]
    l = price_hist["low"]
    c = price_hist["close"]
    if len(c) < period + 1:
        return None
    df = pd.DataFrame({"high": h, "low": l, "close": c})
    atr_series = ta.atr(df["high"], df["low"], df["close"], length=period)
    if atr_series is None or pd.isna(atr_series.iloc[-1]):
        return None
    return float(atr_series.iloc[-1])

def calc_vwap(price_hist: dict) -> float | None:
    """Calculate VWAP approximation from price history"""
    h = price_hist["high"]
    l = price_hist["low"]
    c = price_hist["close"]
    if len(c) < 5:
        return None
    tp_sum = 0.0
    for i in range(len(c)):
        tp = (h[i] + l[i] + c[i]) / 3.0
        tp_sum += tp
    return tp_sum / len(c)

def calc_rsi(prices: list[float], period: int = 14) -> float | None:
    """Calculate RSI from price list"""
    if len(prices) < period + 1:
        return None
    df = pd.DataFrame({"close": prices})
    rsi_series = ta.rsi(df["close"], length=period)
    if rsi_series is None or pd.isna(rsi_series.iloc[-1]):
        return None
    return float(rsi_series.iloc[-1])

def calc_bb(prices: list[float], period: int = 20, std: float = 2.0) -> tuple[float, float, float] | None:
    """Calculate Bollinger Bands (lower, mid, upper)"""
    if len(prices) < period:
        return None
    df = pd.DataFrame({"close": prices})
    bb = ta.bbands(df["close"], length=period, std=std)
    if bb is None:
        return None
    lower = bb.iloc[-1, 0]
    mid = bb.iloc[-1, 1]
    upper = bb.iloc[-1, 2]
    if pd.isna(lower) or pd.isna(mid) or pd.isna(upper):
        return None
    return (float(lower), float(mid), float(upper))

# ============================================
# Morning Momentum Signal
# ============================================
def calc_morning_signal(prices: list[float], strategy_config: dict) -> tuple[str, float]:
    """Morning momentum: EMA crossover + Breakout"""
    if len(prices) < MIN_DATA_POINTS: return ("HOLD", -999.0)
    
    df = pd.DataFrame({"close": prices})
    tf = strategy_config["strategies"]["trend_follow"]["params"]
    ema_short_len = min(tf["ema_short"], len(prices) - 1)
    ema_long_len = min(tf["ema_long"], len(prices) - 1)
    if ema_short_len < 3 or ema_long_len < 3:
        return ("HOLD", -999.0)

    ema_s = ta.ema(df["close"], length=ema_short_len)
    ema_l = ta.ema(df["close"], length=ema_long_len)
    if ema_s is None or ema_l is None: return ("HOLD", -999.0)
    
    score = 0.0
    last = len(df) - 1
    prev = last - 1
    
    if not pd.isna(ema_l.iloc[last]) and not pd.isna(ema_l.iloc[prev]):
        if ema_s.iloc[last] > ema_l.iloc[last] and ema_s.iloc[prev] <= ema_l.iloc[prev]:
            score += 1.0
        elif ema_s.iloc[last] < ema_l.iloc[last] and ema_s.iloc[prev] >= ema_l.iloc[prev]:
            score -= 1.0

    bo = strategy_config["strategies"]["breakout"]["params"]
    period = min(bo["channel_period"], len(prices) - 1)
    if period >= 3 and last >= period:
        high_max = df["close"].iloc[last-period:last].max()
        if df["close"].iloc[last] > high_max:
            score += 1.0
            
    if score >= strategy_config["ensemble"]["buy_threshold"]:
        return ("BUY", score)
    return ("HOLD", score)

# ============================================
# Afternoon Reversal Signal
# ============================================
def calc_afternoon_signal(price_hist: dict, afternoon_cfg: dict) -> tuple[str, dict]:
    """Afternoon reversal: RSI oversold + BB lower + price below VWAP"""
    prices = price_hist["close"]
    if len(prices) < MIN_DATA_POINTS:
        return ("HOLD", {})

    ar = afternoon_cfg["afternoon_reversal"]
    current = prices[-1]

    # RSI
    rsi = calc_rsi(prices, ar.get("rsi_period", 14))
    if rsi is None:
        return ("HOLD", {})

    # Bollinger Bands
    bb = calc_bb(prices, ar.get("bb_period", 20), ar.get("bb_std", 2.0))
    if bb is None:
        return ("HOLD", {})
    bb_lower, bb_mid, bb_upper = bb

    # VWAP
    vwap = calc_vwap(price_hist)

    # Check if price dropped from recent high (reversal candidate)
    recent_high = max(prices[-MIN_DATA_POINTS:])
    recent_low = min(prices[-MIN_DATA_POINTS:])
    move_pct = 0.0
    if recent_high > 0:
        move_pct = ((current - recent_high) / recent_high) * 100

    info = {"rsi": rsi, "bb_lower": bb_lower, "bb_upper": bb_upper, "vwap": vwap, "move_pct": move_pct}

    # BUY: oversold reversal
    rsi_oversold = ar.get("rsi_oversold", 25)
    if rsi <= rsi_oversold and current <= bb_lower:
        return ("BUY", info)

    # SELL: overbought reversal (not used in current strategy, sell_threshold=-99)
    rsi_overbought = ar.get("rsi_overbought", 75)
    if rsi >= rsi_overbought and current >= bb_upper:
        return ("SELL", info)

    return ("HOLD", info)

# ============================================
# Main
# ============================================
def main():
    live_cfg, strat_cfg, afternoon_cfg = load_config()
    auth = KabuAuth(live_cfg["api"]["base_url"], live_cfg["api"]["password"])
    client = KabuClient(live_cfg["api"]["base_url"], auth)
    order_mgr = OrderManager(client, live_cfg)

    price_history = {ticker: {"high": [], "low": [], "close": []} for ticker in live_cfg["watchlist"]}
    last_signal_time = 0
    signal_check_count = 0

    pm_available = afternoon_cfg is not None
    
    print("Bot running... (v12.4 AM Momentum + PM Reversal v1.2)")
    print(f"  Watchlist: {live_cfg['watchlist']}")
    print(f"  [AM] Entry: 9:05-11:00 | threshold: {strat_cfg['ensemble']['buy_threshold']} | SL: {strat_cfg['exit']['stop_loss_atr_multiplier']} ATR / TP: {strat_cfg['exit']['take_profit_rr_ratio']} R:R")
    if pm_available:
        ar = afternoon_cfg["afternoon_reversal"]
        pm_exit = afternoon_cfg["exit"]
        print(f"  [PM] Entry: 12:35-14:00 | RSI<={ar['rsi_oversold']} + BB | SL: {pm_exit['stop_loss_atr_multiplier']} ATR / TP: {pm_exit['take_profit_atr_multiplier']} ATR")
    else:
        print("  [PM] afternoon_config.yaml not found - PM session disabled")
    print(f"  Min data points: {MIN_DATA_POINTS}")

    try:
        while True:
            if not is_market_open():
                time.sleep(30); continue

            # Fetch prices
            for ticker in live_cfg["watchlist"]:
                board = client.get_board(ticker)
                if board and board.get("CurrentPrice"):
                    current_price = float(board["CurrentPrice"])
                    high_price = float(board.get("HighPrice", current_price))
                    low_price = float(board.get("LowPrice", current_price))

                    ph = price_history[ticker]
                    ph["close"].append(current_price)
                    ph["high"].append(high_price)
                    ph["low"].append(low_price)

                    for key in ("high", "low", "close"):
                        if len(ph[key]) > 500:
                            ph[key] = ph[key][-500:]
                time.sleep(1)

            # Position management (works for both AM and PM positions)
            for pos in list(order_mgr.positions):
                ph = price_history.get(pos.ticker)
                if ph is None or len(ph["close"]) < 2:
                    continue
                current = ph["close"][-1]
                current_high = ph["high"][-1]
                trail_mult = strat_cfg["exit"]["trailing_atr_multiplier"]
                atr_period = strat_cfg["exit"].get("atr_period", 14)

                atr_val = calc_atr(ph, atr_period)
                if atr_val is None:
                    atr_val = abs(current - ph["close"][-2])
                    if atr_val == 0:
                        atr_val = 1

                # Accelerated trailing stop (v12.4)
                if current > pos.entry_price:
                    profit_ratio = (current_high - pos.entry_price) / pos.entry_price
                    accel = max(0.7, 1.0 - profit_ratio * 2.0)
                    new_trail = current_high - (atr_val * trail_mult * accel)
                    if new_trail > pos.trailing_stop:
                        pos.trailing_stop = new_trail

                # Exit check
                if current <= pos.stop_loss or current >= pos.take_profit or current <= pos.trailing_stop:
                    order_mgr.exit(pos, current, "Exit triggered")

            # Signal check
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
                            data_len = len(price_history[ticker]["close"])
                            last_price = price_history[ticker]["close"][-1] if data_len > 0 else 0
                            vwap = calc_vwap(price_history[ticker])
                            vwap_str = f"{vwap:.1f}" if vwap else "N/A"
                            ready = "OK" if data_len >= MIN_DATA_POINTS else "waiting"
                            rsi = calc_rsi(price_history[ticker]["close"], 14)
                            rsi_str = f"{rsi:.1f}" if rsi else "N/A"
                            print(f"    {ticker}: price={last_price:.0f} vwap={vwap_str} rsi={rsi_str} data={data_len}/{MIN_DATA_POINTS} [{ready}]")

                    for ticker in live_cfg["watchlist"]:
                        if not order_mgr.can_entry(ticker): continue
                        
                        # ========== MORNING MOMENTUM ==========
                        if morning:
                            prices = price_history[ticker]["close"]
                            signal, score = calc_morning_signal(prices, strat_cfg)

                            if score > 0:
                                print(f"  [{now_str}] [AM] {ticker}: score={score:.1f} signal={signal} (threshold={strat_cfg['ensemble']['buy_threshold']})")

                            if signal == "BUY":
                                current = prices[-1]
                                vwap = calc_vwap(price_history[ticker])
                                if vwap is not None and current < vwap:
                                    print(f"  [{now_str}] [AM] {ticker}: BUY blocked by VWAP (price={current:.0f} < vwap={vwap:.0f})")
                                    continue
                                
                                atr_val = calc_atr(price_history[ticker], strat_cfg["exit"].get("atr_period", 14))
                                if atr_val is None:
                                    atr_val = max(current * 0.01, 1)

                                sl_dist = atr_val * strat_cfg["exit"]["stop_loss_atr_multiplier"]
                                sl_dist = max(sl_dist, 1)
                                risk_per = live_cfg["trade"]["risk_per_trade"]

                                size = int((live_cfg["trade"]["initial_capital"] * risk_per) / sl_dist)
                                size = max((size // 100) * 100, 100)

                                entry_price = current
                                stop_loss = entry_price - sl_dist
                                take_profit = entry_price + sl_dist * strat_cfg["exit"]["take_profit_rr_ratio"]

                                print(f"  [{now_str}] [AM] >>> ENTRY {ticker}: price={entry_price:.0f} SL={stop_loss:.0f} TP={take_profit:.0f} size={size}")
                                order_mgr.entry(ticker, "BUY", entry_price, size, stop_loss, take_profit)

                        # ========== AFTERNOON REVERSAL ==========
                        if afternoon:
                            signal, info = calc_afternoon_signal(price_history[ticker], afternoon_cfg)
                            rsi_val = info.get("rsi", 0)
                            bb_lower = info.get("bb_lower", 0)

                            if rsi_val and rsi_val < 40:
                                print(f"  [{now_str}] [PM] {ticker}: RSI={rsi_val:.1f} signal={signal}")

                            if signal == "BUY":
                                current = price_history[ticker]["close"][-1]

                                pm_exit = afternoon_cfg["exit"]
                                atr_val = calc_atr(price_history[ticker], pm_exit.get("atr_period", 14))
                                if atr_val is None:
                                    atr_val = max(current * 0.01, 1)

                                sl_dist = atr_val * pm_exit["stop_loss_atr_multiplier"]
                                sl_dist = max(sl_dist, 1)
                                tp_dist = atr_val * pm_exit["take_profit_atr_multiplier"]

                                pm_global = afternoon_cfg["global"]
                                risk_per = pm_global["risk_per_trade"]
                                size = int((pm_global["initial_capital"] * risk_per) / sl_dist)
                                size = max((size // 100) * 100, 100)

                                entry_price = current
                                stop_loss = entry_price - sl_dist
                                take_profit = entry_price + tp_dist

                                print(f"  [{now_str}] [PM] >>> ENTRY {ticker}: price={entry_price:.0f} SL={stop_loss:.0f} TP={take_profit:.0f} size={size} RSI={rsi_val:.1f}")
                                order_mgr.entry(ticker, "BUY", entry_price, size, stop_loss, take_profit)

            time.sleep(live_cfg["interval"]["price_check_sec"])

    except KeyboardInterrupt:
        print("\nStopped.")
        print(f"  Total signal checks: {signal_check_count}")
        print(f"  Open positions: {len(order_mgr.positions)}")

if __name__ == "__main__":
    main()
