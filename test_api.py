"""
kabu STATION API 接続テスト

テスト内容:
  1. トークン取得（認証）
  2. 銘柄情報取得（トヨタ自動車）
  3. 板情報取得
  4. 信用取引余力照会
  5. 注文一覧照会

※ 検証用ポート(18081)を使用するため、実際の発注は行いません。
"""

import requests
import json

# ============================================
# 設定（APIパスワードだけ書き換えてください）
# ============================================
API_PASSWORD = "179825519"   # ← ここを書き換える

# 本番ポート（実データ取得）
BASE_URL = "http://localhost:18080/kabusapi"

# ============================================
# テスト実行
# ============================================

def test_connection():
    print("=" * 60)
    print("  kabu STATION API 接続テスト")
    print("=" * 60)

    # --- Test 1: トークン取得 ---
    print("\n■ Test 1: トークン取得（認証）")
    print("-" * 40)

    token = None
    try:
        res = requests.post(
            f"{BASE_URL}/token",
            json={"APIPassword": API_PASSWORD},
            timeout=5,
        )
        print(f"  ステータスコード: {res.status_code}")

        if res.status_code == 200:
            data = res.json()
            token = data.get("Token")
            print(f"  ✅ トークン取得成功!")
            print(f"  Token: {token[:10]}...（セキュリティのため省略）")
        else:
            print(f"  ❌ エラー: {res.text}")
            print("\n  考えられる原因:")
            print("    - APIパスワードが間違っている")
            print("    - kabu STATION が起動していない")
            print("    - kabu STATION にログインしていない")
            return

    except requests.ConnectionError:
        print("  ❌ 接続エラー: kabu STATION に接続できません")
        print("\n  考えられる原因:")
        print("    - kabu STATION が起動していない")
        print("    - kabu STATION にログインしていない")
        print("    - ファイアウォールでブロックされている")
        print("\n  → kabu STATION を起動・ログインしてから再実行してください")
        return

    except Exception as e:
        print(f"  ❌ 予期せぬエラー: {e}")
        return

    # 以降のリクエストで使うヘッダー
    headers = {"X-API-KEY": token}

    # --- Test 2: 銘柄情報取得（トヨタ: 7203, 東証: 1） ---
    print("\n■ Test 2: 銘柄情報取得（7203 トヨタ自動車）")
    print("-" * 40)

    try:
        res = requests.get(
            f"{BASE_URL}/symbol/7203@1",
            headers=headers,
            timeout=5,
        )
        print(f"  ステータスコード: {res.status_code}")

        if res.status_code == 200:
            data = res.json()
            print(f"  ✅ 銘柄情報取得成功!")
            print(f"  銘柄名:     {data.get('DisplayName', 'N/A')}")
            print(f"  市場:       {data.get('Exchange', 'N/A')}")
            print(f"  業種:       {data.get('BisCategory', 'N/A')}")
            print(f"  売買単位:   {data.get('TradingUnit', 'N/A')}株")
            print(f"  前日終値:   {data.get('PreviousClose', 'N/A')}円")
            print(f"  時価総額:   {data.get('TotalMarketValue', 'N/A')}")
        else:
            print(f"  ⚠️ レスポンス: {res.text[:200]}")

    except Exception as e:
        print(f"  ❌ エラー: {e}")

    # --- Test 3: 板情報取得 ---
    print("\n■ Test 3: 板情報取得（7203 トヨタ自動車）")
    print("-" * 40)

    try:
        res = requests.get(
            f"{BASE_URL}/board/7203@1",
            headers=headers,
            timeout=5,
        )
        print(f"  ステータスコード: {res.status_code}")

        if res.status_code == 200:
            data = res.json()
            print(f"  ✅ 板情報取得成功!")
            print(f"  銘柄名:     {data.get('SymbolName', 'N/A')}")
            print(f"  現在値:     {data.get('CurrentPrice', 'N/A')}円")
            print(f"  前日比:     {data.get('ChangePreviousClose', 'N/A')}円")
            print(f"  出来高:     {data.get('TradingVolume', 'N/A')}株")
            print(f"  始値:       {data.get('OpeningPrice', 'N/A')}円")
            print(f"  高値:       {data.get('HighPrice', 'N/A')}円")
            print(f"  安値:       {data.get('LowPrice', 'N/A')}円")

            # 板（気配値）表示
            sell1 = data.get("Sell1", {})
            buy1 = data.get("Buy1", {})
            if sell1 or buy1:
                print(f"\n  【板情報】")
                print(f"  売気配: {sell1.get('Price', 'N/A')}円 × {sell1.get('Qty', 'N/A')}株")
                print(f"  買気配: {buy1.get('Price', 'N/A')}円 × {buy1.get('Qty', 'N/A')}株")
        else:
            print(f"  ⚠️ レスポンス: {res.text[:200]}")

    except Exception as e:
        print(f"  ❌ エラー: {e}")

    # --- Test 4: 信用取引余力照会 ---
    print("\n■ Test 4: 信用取引余力照会")
    print("-" * 40)

    try:
        res = requests.get(
            f"{BASE_URL}/wallet/margin",
            headers=headers,
            timeout=5,
        )
        print(f"  ステータスコード: {res.status_code}")

        if res.status_code == 200:
            data = res.json()
            print(f"  ✅ 信用余力取得成功!")
            margin_buying = data.get("MarginAccountWallet")
            if margin_buying is not None:
                print(f"  信用建余力:     {margin_buying:,.0f}円")
            else:
                print(f"  レスポンス内容:")
                for key, val in data.items():
                    print(f"    {key}: {val}")
        else:
            print(f"  ⚠️ レスポンス: {res.text[:200]}")

    except Exception as e:
        print(f"  ❌ エラー: {e}")

    # --- Test 5: 注文一覧照会 ---
    print("\n■ Test 5: 注文一覧照会")
    print("-" * 40)

    try:
        res = requests.get(
            f"{BASE_URL}/orders",
            headers=headers,
            timeout=5,
        )
        print(f"  ステータスコード: {res.status_code}")

        if res.status_code == 200:
            data = res.json()
            print(f"  ✅ 注文一覧取得成功!")
            if isinstance(data, list):
                print(f"  注文件数: {len(data)}件")
                if len(data) > 0:
                    for order in data[:3]:
                        print(f"    注文ID: {order.get('ID', 'N/A')}")
                        print(f"    状態:   {order.get('State', 'N/A')}")
                        print(f"    銘柄:   {order.get('Symbol', 'N/A')}")
                        print(f"    ---")
                else:
                    print("  （現在注文はありません）")
            else:
                print(f"  レスポンス: {json.dumps(data, indent=2, ensure_ascii=False)[:300]}")
        else:
            print(f"  ⚠️ レスポンス: {res.text[:200]}")

    except Exception as e:
        print(f"  ❌ エラー: {e}")

    # --- まとめ ---
    print("\n" + "=" * 60)
    print("  接続テスト完了!")
    print("=" * 60)
    print(f"\n  使用ポート: 18081（検証用）")
    print(f"  ※ 本番ポートは 18080 です")
    print(f"  ※ 本番への切り替えは自動売買Bot構築時に行います")


if __name__ == "__main__":
    test_connection()