"""
Minimal Spot (Cash) Buy Test — 4063 market order

Purpose:
  Find the correct DelivType + FundType combination for cash buy orders.
  Verified: manual order via kabu STATION works (4063, market, specific account).

Tests all DelivType x FundType combinations and stops on first success.

WARNING:
  REAL orders on PRODUCTION port. Cancel via kabu STATION if accepted.
"""

import requests
import json
import sys
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
    (2, "  ", "DelivType=2(預り金) FundType='  '(自動)"),
    (2, "AA", "DelivType=2(預り金) FundType='AA'(信用代用)"),
    (2, "02", "DelivType=2(預り金) FundType='02'(保護預り)"),
    (0, "  ", "DelivType=0(自動)   FundType='  '(自動)"),
    (0, "AA", "DelivType=0(自動)   FundType='AA'(信用代用)"),
    (0, "02", "DelivType=0(自動)   FundType='02'(保護預り)"),
]


def main():
    print("=" * 60)
    print("  Minimal Spot Buy Test: 4063 (信越化学工業)")
    print("=" * 60)

    # --- Token ---
    try:
        res = requests.post(
            f"{BASE_URL}/token",
            json={"APIPassword": API_PASSWORD},
            timeout=10,
        )
        token = res.json().get("Token") if res.status_code == 200 else None
    except Exception as e:
        print(f"  ❌ Token error: {e}")
        sys.exit(1)

    if not token:
        print("  ❌ Token failed.")
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

        # Log without password
        print(f"  [{i}/{len(COMBOS)}] {desc}")

        try:
            res = requests.post(
                f"{BASE_URL}/sendorder",
                headers={**headers, "Content-Type": "application/json"},
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
            print(f"     FundType  = \"{fund}\"")
            print(f"\n  ⚠️  REAL ORDER PLACED! Cancel via kabu STATION immediately!")
            return
        else:
            print(f"         ❌ Code={code} {msg}\n")

    # All failed
    print("  ❌ All combinations failed.")
    print("     Check account balance or try during market hours.")

    print("\n" + "=" * 60)
    print("  Test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
