"""
JP Stock Auto Trading Bot v12.4 (Accel TS + ATR + VWAP Filter + Debug Log)
"""

import time
import yaml
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from core.auth import KabuAuth
from core.api_client import KabuClient
from core.order_manager import OrderManager, LivePosition

def load_config():
    with open("config/live_config.yaml", "r", encoding="utf-8") as f:
        live = yaml.safe_load(f)
    with open("config/strategy_config.yaml", "r", encoding="utf-8") as f:
        strategy = yaml.safe_load(f)
    return live, strategy

def is_market_open() -> bool:
    now = datetime.now()
    if now.weekday() >= 5: return False
    t = now.hour * 100 + now.minute
    return 900 <= t <= 1525

def is_entry_allowed() -> bool:
    """v12.4: 9:05-11:00 only"""
    now = datetime.now()
    t = now.hour * 100 + now.minute
    return 905 <= t <= 1100

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
    """Calculate VWAP from price history (volume approximated by tick count)"""
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

def calc_signals_with_score(prices: list[float], strategy_config: dict) -> tuple[str, float]:
    """Return (signal, score) for debug logging"""
    if len(prices) < 30: return ("HOLD", -999.0)
    
    df = pd.DataFrame({"close": prices})
    
    # Trend follow
    tf = strategy_config["strategies"]["trend_follow"]["params"]
    ema_s = ta.ema(df["close"], length=tf["ema_short"])
    ema_l = ta.ema(df["close"], length=tf["ema_long"])
    
    if ema_s is None or ema_l is None: return ("HOLD", -999.0)
    
    score = 0.0
    last = len(df) - 1
    prev = last - 1
    
    if not pd.isna(ema_l.iloc[last]) and not pd.isna(ema_l.iloc[prev]):
        if ema_s.iloc[last] > ema_l.iloc[last] and ema_s.iloc[prev] <= ema_l.iloc[prev]:
            score += 1.0
        elif ema_s.iloc[last] < ema_l.iloc[last] and ema_s.iloc[prev] >= ema_l.iloc[prev]:
            score -= 1.0

    # Breakout
    bo = strategy_config["strategies"]["breakout"]["params"]
    period = bo["channel_period"]
    if last >= period:
        high_max = df["close"].iloc[last-period:last].max()
        if df["close"].iloc[last] > high_max:
            score += 1.0
            
    if score >= strategy_config["ensemble"]["buy_threshold"]:
        return ("BUY", score)
    return ("HOLD", score)

def main():
    live_cfg, strat_cfg = load_config()
    auth = KabuAuth(live_cfg["api"]["base_url"], live_cfg["api"]["password"])
    client = KabuClient(live_cfg["api"]["base_url"], auth)
    order_mgr = OrderManager(client, live_cfg)

    price_history = {ticker: {"high": [], "low": [], "close": []} for ticker in live_cfg["watchlist"]}
    last_signal_time = 0
    signal_check_count = 0
    
    print("Bot running... (v12.4 Accel TS + ATR + VWAP Filter + Debug)")
    print(f"  Watchlist: {live_cfg['watchlist']}")
    print(f"  Buy threshold: {strat_cfg['ensemble']['buy_threshold']}")
    print(f"  Entry window: 9:05 - 11:00")
    print(f"  SL: {strat_cfg['exit']['stop_loss_atr_multiplier']} ATR / TP: {strat_cfg['exit']['take_profit_rr_ratio']} R:R")

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

            # Position management
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
                signal_check_count += 1

                if not is_entry_allowed():
                    if signal_check_count % 10 == 1:
                        print(f"  [{now_str}] Entry window closed (9:05-11:00). Positions: {len(order_mgr.positions)}")
                else:
                    # Log data status every 5 checks
                    if signal_check_count % 5 == 1:
                        print(f"\n  [{now_str}] === Signal Check #{signal_check_count} ===")
                        for ticker in live_cfg["watchlist"]:
                            data_len = len(price_history[ticker]["close"])
                            last_price = price_history[ticker]["close"][-1] if data_len > 0 else 0
                            vwap = calc_vwap(price_history[ticker])
                            vwap_str = f"{vwap:.1f}" if vwap else "N/A"
                            print(f"    {ticker}: price={last_price:.0f} vwap={vwap_str} data={data_len}/30")

                    for ticker in live_cfg["watchlist"]:
                        if not order_mgr.can_entry(ticker): continue
                        
                        prices = price_history[ticker]["close"]
                        signal, score = calc_signals_with_score(prices, strat_cfg)

                        # Log BUY signals and near-BUY scores
                        if score > 0:
                            print(f"  [{now_str}] {ticker}: score={score:.1f} signal={signal} (threshold={strat_cfg['ensemble']['buy_threshold']})")

                        if signal == "BUY":
                            current = prices[-1]

                            # VWAP filter
                            vwap = calc_vwap(price_history[ticker])
                            if vwap is not None and current < vwap:
                                print(f"  [{now_str}] {ticker}: BUY blocked by VWAP filter (price={current:.0f} < vwap={vwap:.0f})")
                                continue
                            
                            # ATR-based SL/TP
                            atr_val = calc_atr(price_history[ticker], strat_cfg["exit"].get("atr_period", 14))
                            if atr_val is None:
                                atr_val = max(current * 0.01, 1)

                            sl_mult = strat_cfg["exit"]["stop_loss_atr_multiplier"]
                            tp_rr = strat_cfg["exit"]["take_profit_rr_ratio"]
                            risk_per = live_cfg["trade"]["risk_per_trade"]

                            sl_dist = atr_val * sl_mult
                            sl_dist = max(sl_dist, 1)

                            size = int((live_cfg["trade"]["initial_capital"] * risk_per) / sl_dist)
                            size = max((size // 100) * 100, 100)

                            entry_price = current
                            stop_loss = entry_price - sl_dist
                            take_profit = entry_price + sl_dist * tp_rr

                            print(f"  [{now_str}] >>> ENTRY {ticker}: price={entry_price:.0f} SL={stop_loss:.0f} TP={take_profit:.0f} size={size} ATR={atr_val:.1f}")
                            order_mgr.entry(ticker, "BUY", entry_price, size, stop_loss, take_profit)

            time.sleep(live_cfg["interval"]["price_check_sec"])

    except KeyboardInterrupt:
        print("\nStopped.")
        print(f"  Total signal checks: {signal_check_count}")
        print(f"  Open positions: {len(order_mgr.positions)}")

if __name__ == "__main__":
    main()
