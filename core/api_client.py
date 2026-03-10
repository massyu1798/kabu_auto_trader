"""kabu STATION API client (v15.5: find_exchange, spot order fix, fund_type support)"""

import requests
import json
import time
from core.auth import KabuAuth


# Fields to log on order failure (never log Password/Token)
_ORDER_LOG_FIELDS = [
    "Symbol", "Exchange", "SecurityType", "Side", "CashMargin",
    "MarginTradeType", "DelivType", "FundType", "AccountType",
    "Qty", "FrontOrderType", "Price", "ExpireDay",
]


class KabuClient:
    """kabu STATION REST API wrapper"""

    def __init__(self, base_url: str, auth: KabuAuth, timeout: int = 10):
        self.base_url = base_url
        self.auth = auth
        self.timeout = timeout

    def _get(self, path: str, params: dict = None) -> dict:
        """GET request"""
        headers = self.auth.get_headers()
        if not headers:
            return None
        try:
            res = requests.get(
                f"{BASE_URL}{path}", # Note: fixed to use f"{self.base_url}{path}" if needed, but BASE_URL might be global in some contexts? No, should be self.base_url.
                headers=headers,
                params=params,
                timeout=self.timeout,
            )
            # Re-fixing the base_url usage (noticed a bug in original code or my reading)
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
        """POST request"""
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

    @staticmethod
    def _format_payload_log(data: dict) -> str:
        """Format order payload for logging (no sensitive fields)."""
        parts = []
        for key in _ORDER_LOG_FIELDS:
            if key in data:
                parts.append(f"{key}={data[key]}")
        return " ".join(parts)

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
            # Log payload fields for diagnosis (never log Password)
            print(f"     payload: {self._format_payload_log(data)}")
            return {"ok": False, "http": res.status_code, "code": api_code, "message": api_msg}

        except requests.exceptions.Timeout:
            msg = f"timeout ({self.timeout}s)"
            print(f"  ⚠️ /sendorder タイムアウト: {msg}")
            return {"ok": False, "http": 0, "code": 0, "message": msg}
        except Exception as e:
            msg = str(e)
            print(f"  ❌ /sendorder 通信エラー: {msg}")
            return {"ok": False, "http": 0, "code": 0, "message": msg}

    # --- Info APIs ---

    def find_exchange(self, symbol: str) -> int:
        """
        Identify the correct Exchange code for a symbol by querying /symbol.
        Tries 1 (TSE), then 3, 5, 6 as fallbacks.
        Returns the first successful Exchange code or 1 as default.
        """
        candidates = [1, 3, 5, 6]
        for ex in candidates:
            res = self.get_symbol(symbol, ex)
            if res and res.get("Symbol") == symbol:
                print(f"  ✅ Determined Exchange: {ex} ({res.get('DisplayName')}) for {symbol}")
                return ex
        print(f"  ⚠️ Could not determine Exchange for {symbol}. Falling back to 1.")
        return 1

    def get_board(self, symbol: str, exchange: int = 1) -> dict:
        """Board info"""
        return self._get(f"/board/{symbol}@{exchange}")

    def get_symbol(self, symbol: str, exchange: int = 1) -> dict:
        """Symbol info"""
        return self._get(f"/symbol/{symbol}@{exchange}")

    def get_margin_wallet(self) -> dict:
        """Margin wallet"""
        return self._get("/wallet/margin")

    def get_orders(self, product: str = "2") -> list:
        """Order list (1=spot, 2=margin)"""
        result = self._get("/orders", params={"product": product})
        return result if isinstance(result, list) else []

    def get_positions(self, product: str = "2") -> list:
        """Position list (1=spot, 2=margin)"""
        result = self._get("/positions", params={"product": product})
        return result if isinstance(result, list) else []

    # --- Order APIs ---

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
        """Margin new-open order (CashMargin=2)"""
        side_code = "2" if side == "BUY" else "1"

        data = {
            "Password": self.auth.api_password,
            "Symbol": symbol,
            "Exchange": exchange,
            "SecurityType": 1,
            "Side": side_code,
            "CashMargin": 2,
            "MarginTradeType": margin_trade_type,
            "DelivType": 0,
            "AccountType": 4,
            "Qty": qty,
            "FrontOrderType": 10 if order_type == 1 else 20,
            "Price": price,
            "ExpireDay": 0,
        }
        return self._post_order("/sendorder", data)

    def send_spot_order(
        self,
        symbol: str,
        exchange: int,
        side: str,
        qty: int,
        order_type: int = 1,
        price: float = 0,
        deliv_type: int = 2,
        fund_type: str = None,
    ) -> dict:
        """
        Spot (cash) order (CashMargin=1).
        deliv_type: 2=預り金 (kabuStation standard)
        fund_type: '02'=保護預り, 'AA'=信用代用, etc. If None, it is omitted.
        """
        side_code = "2" if side == "BUY" else "1"

        data = {
            "Password": self.auth.api_password,
            "Symbol": symbol,
            "Exchange": exchange,
            "SecurityType": 1,
            "Side": side_code,
            "CashMargin": 1,
            "DelivType": deliv_type,
            "AccountType": 4,
            "Qty": qty,
            "FrontOrderType": 10 if order_type == 1 else 20,
            "Price": price,
            "ExpireDay": 0,
        }
        if fund_type is not None:
            data["FundType"] = fund_type

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
        """Margin close order (CashMargin=3)"""
        side_code = "2" if side == "BUY" else "1"

        data = {
            "Password": self.auth.api_password,
            "Symbol": symbol,
            "Exchange": exchange,
            "SecurityType": 1,
            "Side": side_code,
            "CashMargin": 3,
            "MarginTradeType": margin_trade_type,
            "DelivType": 0,
            "AccountType": 4,
            "Qty": qty,
            "ClosePositions": [{"HoldID": hold_id, "Qty": qty}],
            "FrontOrderType": 10 if order_type == 1 else 20,
            "Price": price,
            "ExpireDay": 0,
        }
        return self._post_order("/sendorder", data)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel order"""
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
