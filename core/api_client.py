"""kabu STATION API クライアント (v13.1: structured order result)"""

import requests
import json
import time
from core.auth import KabuAuth


class KabuClient:
    """kabu STATION REST API ラッパー"""

    def __init__(self, base_url: str, auth: KabuAuth, timeout: int = 10):
        self.base_url = base_url
        self.auth = auth
        self.timeout = timeout

    def _get(self, path: str, params: dict = None) -> dict:
        """GETリクエスト"""
        headers = self.auth.get_headers()
        if not headers:
            return None
        try:
            res = requests.get(
                f"{self.base_url}{path}",
                headers=headers,
                params=params,
                timeout=self.timeout,
            )
            if res.status_code == 200:
                return res.json()
            elif res.status_code == 401:
                print("  ⚠️ トークン期限切れ。再取得します...")
                self.auth.refresh_token()
                headers = self.auth.get_headers()
                res = requests.get(
                    f"{self.base_url}{path}",
                    headers=headers,
                    params=params,
                    timeout=self.timeout,
                )
                if res.status_code == 200:
                    return res.json()
            print(f"  ⚠️ API エラー: {path} -> {res.status_code}")
            return None
        except requests.exceptions.Timeout:
            print(f"  ⚠️ API タイムアウト: {path} (timeout={self.timeout}s) - skip to next loop")
            return None
        except Exception as e:
            print(f"  ❌ API 通信エラー: {path} -> {e}")
            return None

    def _post(self, path: str, data: dict) -> dict:
        """POSTリクエスト"""
        headers = self.auth.get_headers()
        if not headers:
            return None
        try:
            res = requests.post(
                f"{self.base_url}{path}",
                headers=headers,
                json=data,
                timeout=self.timeout,
            )
            if res.status_code == 200:
                return res.json()
            print(f"  ⚠️ API エラー: {path} -> {res.status_code} {res.text[:200]}")
            return None
        except requests.exceptions.Timeout:
            print(f"  ⚠️ API タイムアウト: {path} (timeout={self.timeout}s)")
            return None
        except Exception as e:
            print(f"  ❌ API 通信エラー: {path} -> {e}")
            return None

    def _post_order(self, path: str, data: dict) -> dict:
        """POST for /sendorder — returns structured result for error handling.

        Success: {"ok": True, "order_id": "...", "raw": <response_json>}
        Failure: {"ok": False, "http": <status_code>, "code": <api_code>, "message": "..."}
        Timeout/Network: {"ok": False, "http": 0, "code": 0, "message": "..."}
        """
        headers = self.auth.get_headers()
        if not headers:
            return {"ok": False, "http": 0, "code": 0, "message": "no auth headers"}
        try:
            res = requests.post(
                f"{self.base_url}{path}",
                headers=headers,
                json=data,
                timeout=self.timeout,
            )
            if res.status_code == 200:
                body = res.json()
                return {"ok": True, "order_id": body.get("OrderId", ""), "raw": body}

            # Parse error body
            api_code = 0
            api_msg = ""
            try:
                err_body = res.json()
                api_code = err_body.get("Code", 0)
                api_msg = err_body.get("Message", res.text[:200])
            except Exception:
                api_msg = res.text[:200]

            print(f"  ⚠️ /sendorder -> {res.status_code} Code={api_code} {api_msg}")
            return {"ok": False, "http": res.status_code, "code": api_code, "message": api_msg}

        except requests.exceptions.Timeout:
            msg = f"timeout ({self.timeout}s)"
            print(f"  ⚠️ /sendorder タイムアウト: {msg}")
            return {"ok": False, "http": 0, "code": 0, "message": msg}
        except Exception as e:
            msg = str(e)
            print(f"  ❌ /sendorder 通信エラー: {msg}")
            return {"ok": False, "http": 0, "code": 0, "message": msg}

    # --- 情報取得系 ---

    def get_board(self, symbol: str, exchange: int = 1) -> dict:
        """板情報取得"""
        return self._get(f"/board/{symbol}@{exchange}")

    def get_symbol(self, symbol: str, exchange: int = 1) -> dict:
        """銘柄情報取得"""
        return self._get(f"/symbol/{symbol}@{exchange}")

    def get_margin_wallet(self) -> dict:
        """信用取引余力照会"""
        return self._get("/wallet/margin")

    def get_orders(self, product: str = "2") -> list:
        """注文一覧照会（2=信用）"""
        result = self._get("/orders", params={"product": product})
        return result if isinstance(result, list) else []

    def get_positions(self, product: str = "2") -> list:
        """建玉一覧照会（2=信用）"""
        result = self._get("/positions", params={"product": product})
        return result if isinstance(result, list) else []

    # --- 発注系 ---

    def send_margin_order(
        self,
        symbol: str,
        exchange: int,
        side: str,
        qty: int,
        order_type: int = 1,
        price: float = 0,
        margin_trade_type: int = 3,
    ) -> dict:
        """
        信用新規注文 — returns structured result.

        Success: {"ok": True, "order_id": "...", "raw": {...}}
        Failure: {"ok": False, "http": <status_code>, "code": <api_code>, "message": "..."}
        """
        side_code = "2" if side == "BUY" else "1"

        data = {
            "Password": self.auth.api_password,
            "Symbol": symbol,
            "Exchange": exchange,
            "SecurityType": 1,
            "Side": side_code,
            "CashMargin": 2,              # 2=新規
            "MarginTradeType": margin_trade_type,
            "DelivType": 0,
            "FundType": "  ",
            "AccountType": 4,             # 4=特定
            "Qty": qty,
            "FrontOrderType": 10 if order_type == 1 else 20,
            "Price": price,
            "ExpireDay": 0,               # 0=当日
        }

        return self._post_order("/sendorder", data)

    def send_margin_close(
        self,
        symbol: str,
        exchange: int,
        side: str,
        qty: int,
        hold_id: str,
        order_type: int = 1,
        price: float = 0,
        margin_trade_type: int = 3,
    ) -> dict:
        """
        信用返済注文
        """
        side_code = "2" if side == "BUY" else "1"

        data = {
            "Password": self.auth.api_password,
            "Symbol": symbol,
            "Exchange": exchange,
            "SecurityType": 1,
            "Side": side_code,
            "CashMargin": 3,              # 3=返済
            "MarginTradeType": margin_trade_type,
            "DelivType": 0,
            "FundType": "  ",
            "AccountType": 4,
            "Qty": qty,
            "ClosePositions": [
                {
                    "HoldID": hold_id,
                    "Qty": qty,
                }
            ],
            "FrontOrderType": 10 if order_type == 1 else 20,
            "Price": price,
            "ExpireDay": 0,
        }

        return self._post("/sendorder", data)

    def cancel_order(self, order_id: str) -> dict:
        """注文取消"""
        data = {
            "Password": self.auth.api_password,
            "OrderID": order_id,
        }
        headers = self.auth.get_headers()
        if not headers:
            return None
        try:
            res = requests.put(
                f"{self.base_url}/cancelorder",
                headers=headers,
                json=data,
                timeout=self.timeout,
            )
            if res.status_code == 200:
                return res.json()
            print(f"  ⚠️ 注文取消エラー: {res.status_code} {res.text[:200]}")
            return None
        except Exception as e:
            print(f"  ❌ 注文取消通信エラー: {e}")
            return None
