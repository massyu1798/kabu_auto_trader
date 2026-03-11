"""
Margin New-Open order test — sendorder diagnosis tool (v17.1)

Purpose:
  Determine which margin trade types are available for CashMargin=2 (new open).
  Uses the EXACT same payload structure as api_client.py send_margin_order().

Tests:
  1. MarginTradeType=1 (制度信用)   — the most common margin type
  2. MarginTradeType=3 (デイトレ)   — day-trade margin (your main config)

v17.1 changes:
  - Exchange=27 (東証+) instead of 1 (東証) per issue #1072
  - DelivType=0 (指定なし) for new-open ← ASYMMETRIC with close (=2)
  - FundType='11' (信用取引) instead of '  '
  - AccountType=4 (特定口座)

DelivType asymmetry:
  - 信用新規 (CashMargin=2): DelivType=0
  - 信用返済 (CashMargin=3): DelivType=2 (お預り金)

Interpretation:
  - OrderId returned     → that margin type works
  - Code=100368          → that margin type is blocked
  - Code=100378          → Exchange mismatch (try 9=SOR instead of 27)
  - Other error          → parameter or market issue

WARNING:
  This script places REAL margin orders on PRODUCTION port (18080).
  If an order succeeds, cancel it via kabu STATION immediately.
  Uses smallest practical lot (100 shares) with a cheap symbol.

IMPORTANT — Authentication Instability (401 / Code=4001009):
  This script tests MarginTradeType=1 and MarginTradeType=3 in a SINGLE run,
  sharing ONE token across both requests.

  If you receive HTTP 401 or Code=4001009 on one case but not the other:
    - This does NOT mean that margin type is unavailable.
    - It likely indicates a transient authentication error (token expiry,
      kabu STATION session reset, or API rate limiting).
    - Do NOT conclude "MTT=X is blocked" based on a single 401 result.

  For reliable per-type diagnosis, run each case independently:
    → Use a dedicated single-case script or re-authenticate before each call.
    → This gives cleaner separation between auth errors and order rejections.
"""

import requests
import json
import sys
import yaml

# ============================================
# Load config (password from live_config.yaml)
# ============================================
CONFIG_PATH = "config/live_config.yaml"

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
    API_PASSWORD = _cfg["api"]["password"]
    BASE_URL = _cfg["api"]["base_url"]
except FileNotFoundError:
    print(f"  ❌ {CONFIG_PATH} not found. Copy from live_config.example.yaml first.")
    sys.exit(1)
except KeyError as e:
    print(f"  ❌ Missing key in {CONFIG_PATH}: {e}")
    sys.exit(1)

# Use a lower-priced symbol to minimize margin requirement
# 8306 = MUFG (三菱UFJ) — typically ~1,500-2,500 JPY range
TEST_SYMBOL = "8306"
TEST_EXCHANGE = 27      # 東証+ (required since 2026-02 per issue #1072)
TEST_SIDE = "2"         # "2" = BUY
TEST_QTY = 100          # Minimum lot

# MarginTradeType candidates
MTT_CANDIDATES = [
    (1, "制度信用"),
    (3, "一般信用(デイトレ)"),
]


def get_token():
    """Obtain API token."""
    try:
        res = requests.post(
            f"{BASE_URL}/token",
            json={"APIPassword": API_PASSWORD},
            timeout=10,
        )
        if res.status_code == 200:
            return res.json().get("Token")
        print(f"  ❌ Token failed: {res.status_code} {res.text[:200]}")
        return None
    except Exception as e:
        print(f"  ❌ Token error: {e}")
        return None


def get_current_price(token, symbol):
    """Fetch current price for display."""
    headers = {"X-API-KEY": token}
    try:
        # /board uses Exchange=1 for info queries (not 27)
        res = requests.get(
            f"{BASE_URL}/board/{symbol}@1",
            headers=headers,
            timeout=10,
        )
        if res.status_code == 200:
            data = res.json()
            return data.get("CurrentPrice"), data.get("SymbolName")
    except Exception:
        pass
    return None, None


def send_margin_new_order(token, margin_trade_type):
    """Send margin new-open order — EXACT same structure as api_client.py."""
    headers = {"X-API-KEY": token, "Content-Type": "application/json"}

    # *** This matches api_client.py send_margin_order() exactly ***
    body = {
        "Password": API_PASSWORD,
        "Symbol": TEST_SYMBOL,
        "Exchange": TEST_EXCHANGE,      # 27 = 東証+ (per issue #1072)
        "SecurityType": 1,              # 1 = Stock
        "Side": TEST_SIDE,              # "2" = Buy
        "CashMargin": 2,               # 2 = Margin New Open (信用新規)
        "MarginTradeType": margin_trade_type,
        "DelivType": 0,                # 0 = 指定なし (new-open uses 0; close uses 2)
        "FundType": "11",              # "11" = 信用取引 (v17.1)
        "AccountType": 4,              # 4 = Specific account (特定口座)
        "Qty": TEST_QTY,
        "FrontOrderType": 10,          # 10 = Market order (成行)
        "Price": 0,                    # 0 for market order
        "ExpireDay": 0,                # 0 = Today
    }

    print(f"\n  📤 POST {BASE_URL}/sendorder")
    # Log payload without password
    log_body = {k: v for k, v in body.items() if k != "Password"}
    print(f"     {json.dumps(log_body, ensure_ascii=False)}")

    try:
        res = requests.post(
            f"{BASE_URL}/sendorder",
            headers=headers,
            json=body,
            timeout=15,
        )
        return res
    except requests.Timeout:
        print("  ❌ Timeout (15s)")
        return None
    except requests.ConnectionError:
        print("  ❌ Connection error — is kabu STATION running?")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def main():
    print("=" * 60)
    print("  sendorder Diagnosis: Margin New Open (CashMargin=2)")
    print("  v17.1: Exchange=27, DelivType=0(新規)/2(返済), FundType='11'")
    print("=" * 60)
    print()
    # ---------------------------------------------------------------
    # NOTE: This script shares a single token across MTT=1 and MTT=3.
    # If either case returns 401 or Code=4001009, treat it as an
    # authentication issue — NOT as evidence that the margin type is
    # blocked.  Re-run after restarting kabu STATION, or use a
    # dedicated single-case script for a cleaner diagnosis.
    # ---------------------------------------------------------------
    print(f"  Symbol:  {TEST_SYMBOL}")
    print(f"  Exchange: {TEST_EXCHANGE} (東証+)")
    print(f"  Side:    BUY")
    print(f"  Qty:     {TEST_QTY} shares")
    print(f"  Order:   Market (成行)")
    print(f"  DelivType: 0 (指定なし — 新規注文用)")
    print(f"  FundType:  11 (信用取引)")
    print(f"  Tests:   {len(MTT_CANDIDATES)} MarginTradeType patterns")

    # --- Step 1: Token ---
    print(f"\n■ Step 1: Obtain API token")
    print("-" * 40)
    token = get_token()
    if not token:
        print("  Cannot proceed. Exiting.")
        sys.exit(1)
    print(f"  ✅ Token: {token[:10]}...")

    # --- Step 2: Current price ---
    print(f"\n■ Step 2: Fetch current price of {TEST_SYMBOL}")
    print("-" * 40)
    price, name = get_current_price(token, TEST_SYMBOL)
    if price:
        print(f"  ✅ {name}  price: {price:,.0f} JPY")
        est_margin = price * TEST_QTY * 0.3  # Approx 30% margin
        print(f"  💰 Estimated margin req: ~{est_margin:,.0f} JPY (100 shares)")
    else:
        print(f"  ⚠️  Could not fetch price")

    # --- Step 3: Try each MarginTradeType ---
    print(f"\n■ Step 3: Test margin new-open orders")
    print("=" * 60)
    # NOTE: Both cases below use the same token obtained in Step 1.
    # A 401 on the second case may indicate token expiry, not a
    # margin type restriction.  See module docstring for details.

    results = []  # [(mtt, mtt_label, success, code, message)]

    for mtt, mtt_label in MTT_CANDIDATES:
        print(f"\n  ── MarginTradeType={mtt} ({mtt_label}) ──")

        res = send_margin_new_order(token, mtt)
        if res is None:
            results.append((mtt, mtt_label, False, 0, "No response"))
            continue

        print(f"\n  📥 Status: {res.status_code}")

        # 401 / 4001009: authentication issue, not margin type rejection
        if res.status_code == 401:
            print(f"     ⚠️  HTTP 401 — authentication error.")
            print(f"     This does NOT indicate MTT={mtt} is unavailable.")
            print(f"     Possible causes: token expiry, kabu STATION session reset.")
            print(f"     → Re-run script or use a single-case script for clean diagnosis.")
            results.append((mtt, mtt_label, False, 401, "Auth error (401) — not a margin type rejection"))
            continue

        try:
            data = res.json()
            print(f"     Body: {json.dumps(data, ensure_ascii=False)}")
        except Exception:
            print(f"     Raw: {res.text[:300]}")
            data = {}

        order_id = data.get("OrderId")
        code = data.get("Code", 0)
        message = data.get("Message", "")

        # Code=4001009: token invalid mid-run — same treatment as 401
        if code == 4001009:
            print(f"\n  ⚠️  Code=4001009 — token invalidated during run.")
            print(f"     This is an authentication error, NOT a margin type rejection.")
            print(f"     → Re-authenticate and test MTT={mtt} in isolation.")
            results.append((mtt, mtt_label, False, code, "Auth error (4001009) — not a margin type rejection"))
            continue

        if res.status_code == 200 and order_id:
            print(f"\n  ✅ ORDER ACCEPTED — OrderId: {order_id}")
            print(f"     MTT={mtt} ({mtt_label}) WORKS!")
            print(f"\n  ⚠️  REAL ORDER PLACED! Cancel via kabu STATION immediately!")
            results.append((mtt, mtt_label, True, 0, f"OrderId={order_id}"))
        else:
            print(f"\n  ❌ REJECTED — Code={code} {message}")
            results.append((mtt, mtt_label, False, code, message))

            if code == 100368:
                print(f"     → Code 100368: margin new-open is BLOCKED for MTT={mtt}")
            elif code == 100378:
                print(f"     → Code 100378: Exchange mismatch. Try Exchange=9 (SOR)?")

    # --- Step 4: Summary ---
    print(f"\n\n■ Step 4: Diagnosis Summary")
    print("=" * 60)

    any_success = any(r[2] for r in results)
    any_100368 = any(r[3] == 100368 for r in results)
    any_auth_error = any(r[3] in (401, 4001009) for r in results)

    for mtt, mtt_label, success, code, message in results:
        status = "✅ OK" if success else f"❌ Code={code}"
        print(f"  MTT={mtt} ({mtt_label:12s}): {status}  {message}")

    print()

    if any_auth_error:
        print(f"  ⚠️  One or more cases returned an authentication error (401/4001009).")
        print(f"     These results are INCONCLUSIVE for margin type availability.")
        print(f"     → Re-run after restarting kabu STATION.")
        print(f"     → For reliable diagnosis: test each MTT in a separate run")
        print(f"       (re-authenticate before each case = '1-case-1-auth' pattern).")

    if any_success:
        ok_types = [f"MTT={r[0]}({r[1]})" for r in results if r[2]]
        ng_types = [f"MTT={r[0]}({r[1]})" for r in results if not r[2]]
        print(f"  ✅ Working types: {', '.join(ok_types)}")
        if ng_types:
            print(f"  ❌ Blocked/errored types: {', '.join(ng_types)}")
        print(f"\n  👉 Update live_config.yaml: margin_trade_type to a working value.")
        print(f"     Cancel any test orders via kabu STATION!")
    elif any_100368:
        print(f"  ❌ All tested margin types returned Code=100368 (blocked).")
        print(f"     → Margin new-open via API is restricted on this account.")
        print(f"     → Check auカブコム証券 account settings / API permissions.")
        print(f"     → Contact auカブコム support if needed.")
    elif not any_auth_error:
        print(f"  ❌ All attempts failed with non-100368 errors.")
        print(f"     → Review error codes above for details.")

    print("\n" + "=" * 60)
    print("  Diagnosis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
