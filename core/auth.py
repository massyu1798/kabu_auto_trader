"""kabu STATION API 認証モジュール"""

import requests
import time


class KabuAuth:
    """トークン管理（自動再取得対応）"""

    def __init__(self, base_url: str, api_password: str, timeout: int = 5):
        self.base_url = base_url
        self.api_password = api_password
        self.timeout = timeout
        self.token = None
        self.token_time = None

    def get_token(self) -> str:
        """トークンを取得（既存トークンがあればそのまま返す）"""
        if self.token is not None:
            return self.token
        return self.refresh_token()

    def refresh_token(self) -> str:
        """トークンを再取得"""
        try:
            res = requests.post(
                f"{self.base_url}/token",
                json={"APIPassword": self.api_password},
                timeout=self.timeout,
            )
            if res.status_code == 200:
                self.token = res.json().get("Token")
                self.token_time = time.time()
                print(f"  ✅ トークン取得成功")
                return self.token
            else:
                print(f"  ❌ トークン取得失敗: {res.status_code} {res.text}")
                return None
        except requests.ConnectionError:
            print("  ❌ kabu STATION に接続できません")
            return None
        except Exception as e:
            print(f"  ❌ トークン取得エラー: {e}")
            return None

    def get_headers(self) -> dict:
        """APIリクエスト用ヘッダーを返す"""
        token = self.get_token()
        if token is None:
            return {}
        return {"X-API-KEY": token}