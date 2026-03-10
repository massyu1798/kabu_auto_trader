"""
Cash Buy (spot) order test — sendorder diagnosis tool

Purpose:
  Determine whether the sendorder API is blocked entirely
  or only margin-new-open orders are restricted.

What this script does:
  1. Obtain an API token (authentication)
  2. Fetch the current board price of 4063 (Shin-Etsu Chemical)
  3. Send a CASH BUY order (CashMargin=1) for 100 shares at market price
  4. Print the full response so you can see success or error details

Interpretation:
  - If the order succeeds  → only margin orders (CashMargin=2/3) are blocked
  - If the order fails with the same error (e.g. Code=100368)
      → sendorder API itself is restricted (account-level block)

WARNING:
  This script places a REAL order on the PRODUCTION port (18080).
  Run it ONLY during market hours if you intend to actually execute,
  or outside market hours to just check whether the API accepts the request.
  You can cancel the order manually via kabu STATION if needed.
"""

import requests
import json
import sys

# ============================================
# Configuration
# ============================================
API_PASSWORD = "179825519"
BASE_URL = "http://localhost:18080/kabusapi"

# Order parameters
SYMBOL = "4063"         # Shin-Etsu Chemical
EXCHANGE = 1            # Tokyo Stock Exchange
SIDE = "2"              # "2" = BUY
QTY = 100               # 100 shares (1 trading unit)
CASH_MARGIN = 1         # 1 = Cash (spot) — the key parameter under test


def get_token():
    """Obtain API token."""
    res = requests.post(
        f"{BASE_URL}/token",
        json={"APIPassword": API_PASSWORD},
        timeout=10,
    )
    if res.status_code == 200:
        return res.json().get("Token")
    else:
        print(f"  ❌ Token request failed: {res.status_code}")
        print(f"     {res.text}")
        return None


def get_current_price(token):
    """Fetch the current market price for display purposes."""
    headers = {"X-API-KEY": token}
    try:
        res = requests.get(
            f"{BASE_URL}/board/{SYMBOL}@{EXCHANGE}",
            headers=headers,
            timeout=10,
        )
        if res.status_code == 200:
            data = res.json()
            return data.get("CurrentPrice"), data.get("SymbolName")
        return None, None
    except Exception:
        return None, None


def send_cash_buy_order(token):
    """Send a cash (spot) buy order — CashMargin=1."""
    headers = {"X-API-KEY": token, "Content-Type": "application/json"}

    body = {
        "Password": API_PASSWORD,
        "Symbol": SYMBOL,
        "Exchange": EXCHANGE,
        "SecurityType": 1,      # 1 = Stock
        "Side": SIDE,           # "2" = Buy
        "CashMargin": CASH_MARGIN,  # 1 = Cash (SPOT) — NOT margin
        "DelivType": 2,         # 2 = Cash delivery (for spot buy)
        "FundType": "  ",       # Auto
        "AccountType": 4,       # 4 = Specific account (特定口座)
        "Qty": QTY,
        "FrontOrderType": 10,   # 10 = Market order (成行)
        "Price": 0,             # 0 for market order
        "ExpireDay": 0,         # 0 = Today
    }

    print(f"\n  📤 Sending order request...")
    print(f"     POST {BASE_URL}/sendorder")
    print(f"     Body: {json.dumps(body, indent=6, ensure_ascii=False)}")

    try:
        res = requests.post(
            f"{BASE_URL}/sendorder",
            headers=headers,
            json=body,
            timeout=15,
        )
        return res
    except requests.Timeout:
        print("  ❌ Request timed out (15s)")
        return None
    except requests.ConnectionError:
        print("  ❌ Connection error — is kabu STATION running?")
        return None
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return None


def main():
    print("=" * 60)
    print("  sendorder Diagnosis: Cash Buy (CashMargin=1)")
    print("=" * 60)
    print(f"\n  Target:  {SYMBOL} (Shin-Etsu Chemical)")
    print(f"  Side:    BUY")
    print(f"  Qty:     {QTY} shares")
    print(f"  Type:    CASH (spot) — CashMargin=1")
    print(f"  Order:   Market order")

    # --- Step 1: Token ---
    print(f"\n■ Step 1: Obtain API token")
    print("-" * 40)
    token = get_token()
    if not token:
        print("\n  Cannot proceed without a token. Exiting.")
        sys.exit(1)
    print(f"  ✅ Token obtained: {token[:10]}...")

    # --- Step 2: Current price (informational) ---
    print(f"\n■ Step 2: Fetch current price of {SYMBOL}")
    print("-" * 40)
    price, name = get_current_price(token)
    if price:
        print(f"  ✅ {name}  current price: {price:,.0f} JPY")
        est_cost = price * QTY
        print(f"  💰 Estimated cost: {est_cost:,.0f} JPY ({QTY} shares)")
    else:
        print(f"  ⚠️  Could not fetch price (market may be closed)")

    # --- Step 3: Send the order ---
    print(f"\n■ Step 3: Send CASH BUY order (CashMargin=1)")
    print("-" * 40)

    res = send_cash_buy_order(token)
    if res is None:
        print("\n  ❌ No response received. Check kabu STATION.")
        sys.exit(1)

    print(f"\n  📥 Response:")
    print(f"     Status code: {res.status_code}")

    try:
        data = res.json()
        print(f"     Body: {json.dumps(data, indent=6, ensure_ascii=False)}")
    except Exception:
        print(f"     Raw: {res.text}")
        data = {}

    # --- Step 4: Interpret the result ---
    print(f"\n■ Step 4: Diagnosis")
    print("=" * 60)

    result_code = data.get("Code")
    order_id = data.get("OrderId")
    result_msg = data.get("Message", "")

    if res.status_code == 200 and order_id:
        print(f"  ✅ ORDER ACCEPTED — OrderId: {order_id}")
        print(f"\n  👉 Conclusion:")
        print(f"     Cash (spot) orders work fine.")
        print(f"     The error is specific to MARGIN orders (CashMargin=2 or 3).")
        print(f"     → Likely a margin trading restriction on your account.")
        print(f"\n  ⚠️  NOTE: A real order was placed!")
        print(f"     Cancel it via kabu STATION if you don't want it executed.")
    else:
        print(f"  ❌ ORDER REJECTED — Code: {result_code}")
        print(f"     Message: {result_msg}")

        if result_code == 100368:
            print(f"\n  👉 Conclusion:")
            print(f"     Same error (100368) on cash order too.")
            print(f"     → sendorder API itself is blocked (not just margin).")
            print(f"     → Check account status / API permissions in kabu STATION.")
        else:
            print(f"\n  👉 Conclusion:")
            print(f"     Different error from the margin order issue.")
            print(f"     Review the error code/message above for details.")

    print("\n" + "=" * 60)
    print("  Diagnosis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
