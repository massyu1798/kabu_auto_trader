"""
日本株自動売買Bot v12.4（加速TS + 正式ATR + 11:00エントリー制��）
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
    """v12.4: 9:05〜11:00のみ新規エントリー許可"""
    now = datetime.now()
    t = now.hour * 100 + now.minute
    return 905 <= t <= 1100

def calc_atr(price_hist: dict, period: int = 14) -> float | None:
    """price_historyからpandas_taで正式なATRを��算"""
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

def calc_signals(prices: list[float], strategy_config: dict) -> str:
    if len(prices) < 30: return "HOLD"
    
    df = pd.DataFrame({"close": prices})
    
    # トレンドフォロー
    tf = strategy_config["strategies"]["trend_follow"]["params"]
    ema_s = ta.ema(df["close"], length=tf["ema_short"])
    ema_l = ta.ema(df["close"], length=tf["ema_long"])
    
    if ema_s is None or ema_l is None: return "HOLD"
    
    score = 0.0
    last = len(df) - 1
    prev = last - 1
    
    # NaNチェックを厳格化
    if not pd.isna(ema_l.iloc[last]) and not pd.isna(ema_l.iloc[prev]):
        if ema_s.iloc[last] > ema_l.iloc[last] and ema_s.iloc[prev] <= ema_l.iloc[prev]:
            score += 1.0
        elif ema_s.iloc[last] < ema_l.iloc[last] and ema_s.iloc[prev] >= ema_l.iloc[prev]:
            score -= 1.0

    # ブレイクアウト
    bo = strategy_config["strategies"]["breakout"]["params"]
    period = bo["channel_period"]
    if last >= period:
        high_max = df["close"].iloc[last-period:last].max()
        if df["close"].iloc[last] > high_max:
            score += 1.0
            
    # 統合判定
    if score >= strategy_config["ensemble"]["buy_threshold"]:
        return "BUY"
    return "HOLD"

def main():
    live_cfg, strat_cfg = load_config()
    auth = KabuAuth(live_cfg["api"]["base_url"], live_cfg["api"]["password"])
    client = KabuClient(live_cfg["api"]["base_url"], auth)
    order_mgr = OrderManager(client, live_cfg)

    price_history = {ticker: {"high": [], "low": [], "close": []} for ticker in live_cfg["watchlist"]}
    last_signal_time = 0
    
    print("Bot稼働中... (v12.4 加速TS + 正式ATR)")

    try:
        while True:
            if not is_market_open():
                time.sleep(30); continue

            # 株価取得
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

                    # 最大500件に制限
                    for key in ("high", "low", "close"):
                        if len(ph[key]) > 500:
                            ph[key] = ph[key][-500:]
                time.sleep(1)   # 1銘柄ごとに1秒待つ

            # ポジション管理
            for pos in list(order_mgr.positions):
                ph = price_history.get(pos.ticker)
                if ph is None or len(ph["close"]) < 2:
                    continue
                current = ph["close"][-1]
                current_high = ph["high"][-1]
                trail_mult = strat_cfg["exit"]["trailing_atr_multiplier"]
                atr_period = strat_cfg["exit"].get("atr_period", 14)

                # 正式ATR計算（データ不足時は簡易推定にフォールバック）
                atr_val = calc_atr(ph, atr_period)
                if atr_val is None:
                    atr_val = abs(current - ph["close"][-2])
                    if atr_val == 0:
                        atr_val = 1

                # トレーリング更新（v12.4: 加速TS）
                if current > pos.entry_price:
                    profit_ratio = (current_high - pos.entry_price) / pos.entry_price
                    accel = max(0.7, 1.0 - profit_ratio * 2.0)
                    new_trail = current_high - (atr_val * trail_mult * accel)
                    if new_trail > pos.trailing_stop:
                        pos.trailing_stop = new_trail

                # 決���判定
                if current <= pos.stop_loss or current >= pos.take_profit or current <= pos.trailing_stop:
                    order_mgr.exit(pos, current, "Exit triggered")

            # シグナル判定
            if time.time() - last_signal_time >= live_cfg["interval"]["signal_check_sec"]:
                last_signal_time = time.time()
                if not is_entry_allowed():
                    pass  # 11:00以降は新規エントリーしない
                else:
                    for ticker in live_cfg["watchlist"]:
                        if not order_mgr.can_entry(ticker): continue
                        
                        prices = price_history[ticker]["close"]
                        if calc_signals(prices, strat_cfg) == "BUY":
                            current = prices[-1]
                            
                            # 正式ATRベースのSL/TP計算
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
                            
                            order_mgr.entry(ticker, "BUY", entry_price, size, stop_loss, take_profit)

            time.sleep(live_cfg["interval"]["price_check_sec"])

    except KeyboardInterrupt:
        print("停止しました")

if __name__ == "__main__":
    main()
