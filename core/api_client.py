"""kabu STATION API client (v18.0: spec-aligned margin orders, multi-lot close)"""

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

# Exchange candidates for /symbol info queries
_SYMBOL_EXCHANGE_CANDIDATES = [1, 3, 5, 6]

# Exchange candidates for /sendorder (new-open)
# 27 = 東証+ (required since 2026-02 per issue #1072)
#  9 = SOR   (alternative)
#  1 = 東証  (DEPRECATED for new orders; only for closing Exchange=1 positions)
_ORDER_EXCHANGE_CANDIDATES = [27, 9]

# Error classification map
_ERROR_CATEGORIES = {
    100378: "market_mismatch",
    1010004: "delivtype_invalid",
    100031: "delivtype_invalid",
    4001005: "parameter_convert",
    4001006: "rate_limit",
    100368: "margin_blocked",
    8: "close_position_error",
}


def classify_order_error(code: int) -> str:
    """Classify API error code into a diagnostic category."""
    return _ERROR_CATEGORIES.get(code, "unknown")


class KabuClient:
    """kabu STATION REST API wrapper"""

    def __init__(self, base_url: str, auth: KabuAuth, timeout: int = 10):
        self.base_url = base_url
        self.auth = auth
        self.timeout = timeout

    def _get(self, path: str, params: dict = None) -> dict:
        """GET request (single call with 401 retry)."""
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
                if not headers:
                    return None
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
        Failure: {"ok": False, "http": <status_code>, "code": <api_code>,
                  "message": "...", "category": "..."}
        Timeout/Network: {"ok": False, "http": 0, "code": 0,
                          "message": "...", "category": "network"}
        """
        headers = self.auth.get_headers()
        if not headers:
            return {"ok": False, "http": 0, "code": 0,
                    "message": "no auth headers", "category": "auth"}
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

            category = classify_order_error(api_code)

            print(f"  ⚠️ /sendorder -> {res.status_code} Code={api_code} [{category}] {api_msg}")
            # Log payload fields for diagnosis (never log Password)
            print(f"     payload: {self._format_payload_log(data)}")
            return {"ok": False, "http": res.status_code, "code": api_code,
                    "message": api_msg, "category": category}

        except requests.exceptions.Timeout:
            msg = f"timeout ({self.timeout}s)"
            print(f"  ⚠️ /sendorder タイムアウト: {msg}")
            return {"ok": False, "http": 0, "code": 0,
                    "message": msg, "category": "network"}
        except Exception as e:
            msg = str(e)
            print(f"  ❌ /sendorder 通信エラー: {msg}")
            return {"ok": False, "http": 0, "code": 0,
                    "message": msg, "category": "network"}

    # --- Info APIs ---

    def find_exchange(self, symbol: str) -> int:
        """Identify the correct Exchange code for a symbol by querying /symbol.

        This is for INFO queries (/symbol, /board). Returns 1 (TSE) typically.
        For ORDER Exchange, use find_order_exchange() instead.
        """
        for ex in _SYMBOL_EXCHANGE_CANDIDATES:
            res = self.get_symbol(symbol, ex)
            if res and res.get("Symbol") == symbol:
                display = res.get("DisplayName", "?")
                unit = res.get("TradingUnit", "?")
                resolved_ex = res.get("Exchange", ex)
                print(f"  ✅ Exchange resolved (info): {symbol} -> Exchange={resolved_ex}"
                      f" ({display}) TradingUnit={unit}")
                return resolved_ex
        print(f"  ⚠️ Could not determine Exchange for {symbol}. Falling back to 1.")
        return 1

    def find_order_exchange(self, symbol: str) -> int:
        """Determine the correct Exchange for /sendorder (new-open).

        Per issue #1072 (2026-02~), Exchange=1 is DEPRECATED for new orders.
        New orders require Exchange=27 (東証+) or 9 (SOR).

        Returns: 27 (preferred).
        """
        print(f"  ✅ Order Exchange: defaulting to 27 (東証+) for {symbol}")
        return 27

    def get_board(self, symbol: str, exchange: int = 1) -> dict:
        """Board info"""
        return self._get(f"/board/{symbol}@{exchange}")

    def get_symbol(self, symbol: str, exchange: int = 1) -> dict:
        """Symbol info"""
        return self._get(f"/symbol/{symbol}@{exchange}")

    def get_margin_wallet(self) -> dict:
        """Margin wallet"""
        return self._get("/wallet/margin")

    def get_cash_wallet(self) -> dict:
        """Cash wallet (spot)"""
        return self._get("/wallet/cash")

    def get_orders(self, product: str = "2") -> list:
        """Order list (1=spot, 2=margin)"""
        result = self._get("/orders", params={"product": product})
        return result if isinstance(result, list) else []

    def get_positions(self, product: str = "2") -> list:
        """Position list (1=spot, 2=margin)

        Returns list of position dicts. Each position includes:
        - ExecutionID: the ID to use as HoldID in ClosePositions
        - Symbol, Exchange, Side, Qty, LeavesQty, Price, etc.
        """
        result = self._get("/positions", params={"product": product})
        return result if isinstance(result, list) else []

    def resolve_position_hold_ids(self, symbol: str, side: str) -> list:
        """Fetch open margin positions for a symbol and return hold info.

        After a new-open order fills, call this to get the ExecutionID(s)
        needed for ClosePositions[].HoldID in the close order.
        Returns ALL matching lots (supports split fills / multiple executions).

        Args:
            symbol: e.g. "8306"
            side: "BUY" or "SELL" (the ENTRY side, i.e. Side=2 for BUY)

        Returns:
            list of dicts: [{"hold_id": str, "qty": int, "exchange": int, "price": float}, ...]
            Each hold_id is the ExecutionID from /positions.
        """
        positions = self.get_positions(product="2")
        if not positions:
            return []

        # Side in API: "2"=BUY, "1"=SELL
        target_side = "2" if side == "BUY" else "1"

        results = []
        for pos in positions:
            if str(pos.get("Symbol", "")) != str(symbol):
                continue
            if str(pos.get("Side", "")) != target_side:
                continue

            leaves_qty = pos.get("LeavesQty", 0)
            if leaves_qty <= 0:
                continue

            exec_id = pos.get("ExecutionID", "")
            if not exec_id:
                continue

            results.append({
                "hold_id": str(exec_id),
                "qty": int(leaves_qty),
                "exchange": int(pos.get("Exchange", 1)),
                "price": float(pos.get("Price", 0)),
            })

        return results

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
        deliv_type: int = 0,
        fund_type: str = "11",
    ) -> dict:
        """Margin new-open order (CashMargin=2).

        API spec (current as of 2026-02):
        - Exchange: 27 (東証+) or 9 (SOR). Exchange=1 is DEPRECATED for new orders.
        - DelivType: 0 (指定なし) for margin new-open.
          ※ New-open and close have ASYMMETRIC DelivType rules:
            - 信用新規 (CashMargin=2): DelivType=0
            - 信用返済 (CashMargin=3): DelivType=2 (お預り金)
        - FundType: '11' (信用取引). Spec says optional for margin (can omit),
          but explicit '11' avoids ambiguity.
        - AccountType: 4 (特定口座)
        - MarginTradeType: 3 (一般信用デイトレ)
        """
        side_code = "2" if side == "BUY" else "1"

        data = {
            "Password": self.auth.api_password,
            "Symbol": symbol,
            "Exchange": exchange,
            "SecurityType": 1,
            "Side": side_code,
            "CashMargin": 2,
            "MarginTradeType": margin_trade_type,
            "DelivType": deliv_type,
            "FundType": fund_type,
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
        fund_type: str = "02",
    ) -> dict:
        """Spot (cash) order (CashMargin=1).

        Builds a minimal payload for spot orders. Key rules per API spec:
        - CashMargin=1 (always)
        - Exchange: 27 (東証+) during normal hours, 1 (東証) during maintenance
        - DelivType: 2=預り金 (standard for spot buy)
        - FundType: '02'=保護預り (REQUIRED — omitting causes 4001005)
        - AccountType: 4=特定口座
        - MarginTradeType: NEVER included (spot-only)
        - No empty-string or whitespace-only fields

        Args:
            deliv_type: 2=預り金 (default, standard)
            fund_type:  '02'=保護預り (default, proven working for AccountType=4)
                        'AA'=信用代用
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
            "FundType": fund_type,
            "AccountType": 4,
            "Qty": qty,
            "FrontOrderType": 10 if order_type == 1 else 20,
            "Price": price,
            "ExpireDay": 0,
        }

        # Safety: strip whitespace-only FundType (causes 4001005)
        if not str(data.get("FundType", "")).strip():
            data["FundType"] = "02"

        return self._post_order("/sendorder", data)

    def send_margin_close(
        self,
        symbol: str,
        exchange: int,
        side: str,
        qty: int,
        close_positions: list[dict] = None,
        order_type: int = 1,
        price: float = 0,
        margin_trade_type: int = 3,
        deliv_type: int = 2,
        fund_type: str = "11",
    ) -> dict:
        """Margin close order (CashMargin=3).

        API spec (current as of 2026-02):
        - Exchange: MUST match the Exchange of the position being closed.
          Per issue #1072: Exchange must match the position's Exchange.
          In practice, we unify to Exchange=27 (東証+).
        - DelivType: 2 (お預り金) for margin close.
          ※ ASYMMETRIC with new-open: new-open uses DelivType=0, close uses 2.
        - FundType: '11' (信用取引) — optional per spec but explicit is safer.
        - ClosePositions: list of {"HoldID": str, "Qty": int} dicts.
          HoldID = ExecutionID from /positions.
          Multiple entries are supported for split fills / multiple lots.

        Args:
            close_positions: list of {"HoldID": str, "Qty": int} dicts.
                If None, falls back to a single entry with empty HoldID (not recommended).

        Returns:
            dict: Structured result (same format as _post_order).
        """
        side_code = "2" if side == "BUY" else "1"

        # Build ClosePositions — accept list or fall back to single empty entry
        if close_positions is None:
            close_positions = [{"HoldID": "", "Qty": qty}]

        data = {
            "Password": self.auth.api_password,
            "Symbol": symbol,
            "Exchange": exchange,
            "SecurityType": 1,
            "Side": side_code,
            "CashMargin": 3,
            "MarginTradeType": margin_trade_type,
            "DelivType": deliv_type,
            "FundType": fund_type,
            "AccountType": 4,
            "Qty": qty,
            "ClosePositions": close_positions,
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
