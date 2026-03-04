"""
日本株自動売買Bot（エラー修正・堅牢版）
"""

import time
import yaml
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from core.auth import KabuAuth
from core.api_client import KabuClient
from core.order_manager import OrderManager, LivePosition # 明示的インポート

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

# ★ v12.4 追加: エントリー可能時間帯（9:00〜10:00のみ）
def is_entry_allowed() -> bool:
    now = datetime.now()
    t = now.hour * 100 + now.minute
    return 900 <= t <= 1000

def calc_signals(prices: list[float], strategy_config: dict) -> str:
    if len(prices) < 30: return "HOLD" # 最低30本必要
    
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

    price_history = {ticker: [] for ticker in live_cfg["watchlist"]}
    last_signal_time = 0
    
    print("Bot稼働中...")

    try:
        while True:
            if not is_market_open():
                time.sleep(30); continue

            # 株価取得
            for ticker in live_cfg["watchlist"]:
                board = client.get_board(ticker)
                if board and board.get("CurrentPrice"):
                    price_history[ticker].append(float(board["CurrentPrice"]))
                    if len(price_history[ticker]) > 500:
                        price_history[ticker] = price_history[ticker][-500:]
                time.sleep(1)   # ★ 追加：1銘柄ごとに1秒待つ

            # ポジション管理
            for pos in list(order_mgr.positions):
                prices = price_history.get(pos.ticker, [])
                if len(prices) < 2: continue
                current = prices[-1]
                
                # トレーリング更新（v12.4: 加速TS）
                if current > pos.entry_price:
                    atr_est = abs(current - prices[-2]) if len(prices) >= 2 else 1
                    trail_mult = strat_cfg["exit"]["trailing_atr_multiplier"]

                    # 加速係数: 含み益が大きいほどTSを締める
                    profit_ratio = (current - pos.entry_price) / pos.entry_price
                    accel = max(0.7, 1.0 - profit_ratio * 2.0)

                    new_trail = current - (atr_est * trail_mult * accel)
                    if new_trail > pos.trailing_stop:
                        pos.trailing_stop = new_trail

                # 決済判定
                if current <= pos.stop_loss or current >= pos.take_profit or current <= pos.trailing_stop:
                    order_mgr.exit(pos, current, "Exit triggered")

            # シグナル判定
            if time.time() - last_signal_time >= live_cfg["interval"]["signal_check_sec"]:
                last_signal_time = time.time()
                if not is_entry_allowed():         # ★ 追加: 10:00以降は新規エントリーしない
                    continue                       # ★ 追加
                for ticker in live_cfg["watchlist"]:
                    if not order_mgr.can_entry(ticker): continue
                    
                    prices = price_history[ticker]
                    if calc_signals(prices, strat_cfg) == "BUY":
                        current = prices[-1]
                        # 安全な株数計算
                        sl_dist = max(current * 0.01, 1) # 最低1円
                        size = int((live_cfg["trade"]["initial_capital"] * 0.005) / sl_dist)
                        size = max((size // 100) * 100, 100)
                        
                        order_mgr.entry(ticker, "BUY", current, size, current-sl_dist, current+sl_dist*2)

            time.sleep(live_cfg["interval"]["price_check_sec"])

    except KeyboardInterrupt:
        print("停止しました")

if __name__ == "__main__":
    main()