"""
Cash Buy (spot) order test — sendorder diagnosis tool

Purpose:
  Determine whether the sendorder API is blocked entirely
  or only margin-new-open orders are restricted.

What this script does:
  1. Obtain an API token (authentication)
  2. Fetch the current board price of 4063 (Shin-Etsu Chemical)
  3. Try CASH BUY (CashMargin=1) with multiple FundType values
     to find the correct deposit category for the account
  4. Print the full response and diagnosis for each attempt

FundType candidates (per official API reference):
  "11" = Deposit (お預り金)
  "02" = Custody (保護預り)
  "AA" = Margin collateral (信用代用)

WARNING:
  This script places REAL orders on the PRODUCTION port (18080).
  If an order succeeds, cancel it via kabu STATION if you don't want it executed.
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
CASH_MARGIN = 1         # 1 = Cash (spot)

# FundType candidates to try in order
FUND_TYPE_CANDIDATES = [
    ("11", "Deposit (お預り金)"),
    ("02", "Custody (保護預り)"),
    ("AA", "Margin collateral (信用代用)"),
]


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


def send_cash_buy_order(token, fund_type):
    """Send a cash (spot) buy order with the specified FundType."""
    headers = {"X-API-KEY": token, "Content-Type": "application/json"}

    body = {
        "Password": API_PASSWORD,
        "Symbol": SYMBOL,
        "Exchange": EXCHANGE,
        "SecurityType": 1,          # 1 = Stock
        "Side": SIDE,               # "2" = Buy
        "CashMargin": CASH_MARGIN,  # 1 = Cash (SPOT)
        "DelivType": 2,             # 2 = Cash delivery (預り金)
        "FundType": fund_type,      # Variable — testing multiple values
        "AccountType": 4,           # 4 = Specific account (特定口座)
        "Qty": QTY,
        "FrontOrderType": 10,       # 10 = Market order (成行)
        "Price": 0,                 # 0 for market order
        "ExpireDay": 0,             # 0 = Today
    }

    print(f"\n  📤 Sending order request...")
    print(f"     POST {BASE_URL}/sendorder")
    print(f"     FundType = \"{fund_type}\"")
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
    print("  — FundType auto-detection mode —")
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

    # --- Step 3: Try each FundType ---
    print(f"\n■ Step 3: Try CASH BUY with multiple FundType values")
    print("=" * 60)

    success_found = False

    for i, (fund_type, fund_desc) in enumerate(FUND_TYPE_CANDIDATES, 1):
        print(f"\n  ── Attempt {i}/{len(FUND_TYPE_CANDIDATES)}: "
              f"FundType=\"{fund_type}\" ({fund_desc}) ──")

        res = send_cash_buy_order(token, fund_type)
        if res is None:
            print("  ❌ No response received.")
            continue

        print(f"\n  📥 Response:")
        print(f"     Status code: {res.status_code}")

        try:
            data = res.json()
            print(f"     Body: {json.dumps(data, indent=6, ensure_ascii=False)}")
        except Exception:
            print(f"     Raw: {res.text}")
            data = {}

        result_code = data.get("Code")
        order_id = data.get("OrderId")
        result_msg = data.get("Message", "")

        if res.status_code == 200 and order_id:
            print(f"\n  ✅ ORDER ACCEPTED — OrderId: {order_id}")
            print(f"     FundType=\"{fund_type}\" ({fund_desc}) is CORRECT!")
            print(f"\n  ⚠️  A real order was placed!")
            print(f"     Cancel it via kabu STATION if you don't want it executed.")
            success_found = True
            break  # No need to try more
        else:
            print(f"\n  ❌ REJECTED — Code: {result_code}, Message: {result_msg}")

            # If the error is about the original issue (not FundType), stop immediately
            if result_code == 100368:
                print(f"\n  🚨 Original error (100368) detected!")
                print(f"     → sendorder API itself is blocked (not just margin).")
                print(f"     → No need to try other FundType values.")
                break

    # --- Step 4: Summary ---
    print(f"\n\n■ Step 4: Diagnosis Summary")
    print("=" * 60)

    if success_found:
        print(f"  ✅ Cash (spot) order SUCCEEDED.")
        print(f"     → The original error is specific to MARGIN orders.")
        print(f"     → Likely a margin trading restriction on your account.")
    else:
        last_code = data.get("Code") if data else None
        if last_code == 100368:
            print(f"  ❌ Same error (100368) on cash order too.")
            print(f"     → sendorder API itself is blocked.")
            print(f"     → Check account status / API permissions in kabu STATION.")
        else:
            print(f"  ❌ All FundType attempts were rejected.")
            print(f"     → Last error: Code={last_code}, Message={data.get('Message', 'N/A')}")
            print(f"     → This may be a market-hours issue (休場中) or account config.")
            print(f"     → Try again during market hours (9:00-15:00 on a trading day).")

    print("\n" + "=" * 60)
    print("  Diagnosis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
