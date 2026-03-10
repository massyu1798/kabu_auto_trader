"""
Minimal Spot (Cash) Buy Test — 4063 market order

Purpose:
  Find the correct DelivType + FundType combination for cash buy orders.
  Verified: manual order via kabu STATION works (4063, market, specific account).

Tests all DelivType x FundType combinations with rate-limit delay.
Re-acquires token before each sendorder (kabuS may invalidate token on error).
Stops on first success.

WARNING:
  REAL orders on PRODUCTION port. Cancel via kabu STATION if accepted.
"""

import requests
import json
import sys
import time
import yaml

# ============================================
# Load config
# ============================================
CONFIG_PATH = "config/live_config.yaml"

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
    API_PASSWORD = _cfg["api"]["password"]
    BASE_URL = _cfg["api"]["base_url"]
except FileNotFoundError:
    print(f"  ❌ {CONFIG_PATH} not found.")
    sys.exit(1)
except KeyError as e:
    print(f"  ❌ Missing key in {CONFIG_PATH}: {e}")
    sys.exit(1)

# ============================================
# DelivType x FundType combinations to try
# ============================================
COMBOS = [
    (2, "AA", "DelivType=2(預り金) FundType='AA'(信用代用)"),
    (2, "  ", "DelivType=2(預り金) FundType='  '(自動)"),
    (2, "02", "DelivType=2(預り金) FundType='02'(保護預り)"),
    (0, "AA", "DelivType=0(自動)   FundType='AA'(信用代用)"),
    (0, "02", "DelivType=0(自動)   FundType='02'(保護預り)"),
    (0, "  ", "DelivType=0(自動)   FundType='  '(自動)"),
]

# Delay between requests to avoid 4001006 rate limit
REQUEST_DELAY_SEC = 3.0


def get_token():
    """Obtain a fresh API token."""
    try:
        res = requests.post(
            f"{BASE_URL}/token",
            json={"APIPassword": API_PASSWORD},
            timeout=10,
        )
        if res.status_code == 200:
            return res.json().get("Token")
        print(f"    ⚠️ Token refresh failed: {res.status_code} {res.text[:100]}")
        return None
    except Exception as e:
        print(f"    ⚠️ Token refresh error: {e}")
        return None


def main():
    print("=" * 60)
    print("  Minimal Spot Buy Test: 4063 (信越化学工業)")
    print(f"  Delay between requests: {REQUEST_DELAY_SEC}s")
    print(f"  Token: re-acquired before EACH sendorder")
    print("=" * 60)

    # --- Initial token (for board query only) ---
    token = get_token()
    if not token:
        print("  ❌ Initial token failed.")
        sys.exit(1)
    print(f"  ✅ Token: {token[:10]}...")

    # --- Current price ---
    headers = {"X-API-KEY": token}
    try:
        res = requests.get(f"{BASE_URL}/board/4063@1", headers=headers, timeout=10)
        if res.status_code == 200:
            bd = res.json()
            print(f"  ✅ {bd.get('SymbolName')} price: {bd.get('CurrentPrice'):,.0f} JPY")
    except Exception:
        pass

    # --- Try each combination ---
    print(f"\n  Testing {len(COMBOS)} DelivType x FundType combinations...\n")

    for i, (deliv, fund, desc) in enumerate(COMBOS, 1):

        # Rate limit delay (skip before first request)
        if i > 1:
            print(f"  ⏳ Waiting {REQUEST_DELAY_SEC}s...")
            time.sleep(REQUEST_DELAY_SEC)

        # Re-acquire token before each sendorder
        token = get_token()
        if not token:
            print(f"  [{i}/{len(COMBOS)}] {desc}")
            print(f"         ❌ Could not get token. Skipping.\n")
            continue

        body = {
            "Password": API_PASSWORD,
            "Symbol": "4063",
            "Exchange": 1,
            "SecurityType": 1,
            "Side": "2",
            "CashMargin": 1,
            "DelivType": deliv,
            "FundType": fund,
            "AccountType": 4,
            "Qty": 100,
            "FrontOrderType": 10,
            "Price": 0,
            "ExpireDay": 0,
        }

        print(f"  [{i}/{len(COMBOS)}] {desc}")

        try:
            res = requests.post(
                f"{BASE_URL}/sendorder",
                headers={"X-API-KEY": token, "Content-Type": "application/json"},
                json=body,
                timeout=15,
            )
            data = res.json() if res.text else {}
        except Exception as e:
            print(f"         ❌ Error: {e}\n")
            continue

        code = data.get("Code", 0)
        order_id = data.get("OrderId")
        msg = data.get("Message", "")

        if res.status_code == 200 and order_id:
            print(f"         ✅ SUCCESS — OrderId: {order_id}")
            print(f"\n  🎯 Correct combination found!")
            print(f"     DelivType = {deliv}")
            fund_display = "'  ' (two half-width spaces)" if fund.strip() == "" else f"'{fund}'"
            print(f"     FundType  = {fund_display}")
            print(f"\n  ⚠️  REAL ORDER PLACED! Cancel via kabu STATION immediately!")
            return
        else:
            print(f"         ❌ Code={code} {msg}")

            # If rate limited, wait extra and retry
            if code == 4001006:
                print(f"         ⏳ Rate limited. Waiting 10s and retrying...")
                time.sleep(10)
                token = get_token()
                if not token:
                    print(f"         ❌ Token failed on retry.\n")
                    continue
                try:
                    res2 = requests.post(
                        f"{BASE_URL}/sendorder",
                        headers={"X-API-KEY": token, "Content-Type": "application/json"},
                        json=body,
                        timeout=15,
                    )
                    data2 = res2.json() if res2.text else {}
                except Exception as e2:
                    print(f"         ❌ Retry error: {e2}\n")
                    continue

                code2 = data2.get("Code", 0)
                order_id2 = data2.get("OrderId")
                msg2 = data2.get("Message", "")

                if res2.status_code == 200 and order_id2:
                    print(f"         ✅ RETRY SUCCESS — OrderId: {order_id2}")
                    print(f"\n  🎯 Correct combination found!")
                    print(f"     DelivType = {deliv}")
                    fund_display = "'  ' (two half-width spaces)" if fund.strip() == "" else f"'{fund}'"
                    print(f"     FundType  = {fund_display}")
                    print(f"\n  ⚠️  REAL ORDER PLACED! Cancel via kabu STATION immediately!")
                    return
                else:
                    print(f"         ❌ Retry: Code={code2} {msg2}")

            print()

    # All failed
    print("\n  ❌ All combinations failed.")
    print("     Possible causes:")
    print("     - Insufficient cash balance for spot buy (~610,000 JPY needed)")
    print("     - Account does not allow spot trading via API")
    print("     - Try during market hours if not already")

    print("\n" + "=" * 60)
    print("  Test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
