"""
API データ取得確認用モニター
監視銘柄の現在値を 5秒おきに表示します
"""
import time
import yaml
from core.auth import KabuAuth
from core.api_client import KabuClient

def main():
    with open("config/live_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    auth = KabuAuth(config["api"]["base_url"], config["api"]["password"])
    client = KabuClient(config["api"]["base_url"], auth)

    print("="*50)
    print("📈 リアルタイム価格取得モニター（10秒更新）")
    print("="*50)

    watchlist = config["watchlist"]
    
    try:
        while True:
            print(f"\n--- 取得時刻: {time.strftime('%H:%M:%S')} ---")
            for ticker in watchlist:
                board = client.get_board(ticker)
                if board is None:
                    print("board is None (likely auth failed or API error)")
                else:
                    print("BOARD KEYS:", list(board.keys()))

                if board:
                    name = board.get("SymbolName", "不明")
                    price = board.get("CurrentPrice", "取得失敗")
                    change = board.get("ChangePreviousClose", 0)
                    print(f" [{ticker}] {name:<10}: {price:>8} 円 (前日比 {change:>+5})")
                else:
                    print(f" [{ticker}]: APIからの応答がありません")
            
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n停止しました")

if __name__ == "__main__":
    main()