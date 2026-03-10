"""
Minimal Spot (Cash) Buy Diagnosis Tool (v15.6)

Purpose:
  1. Determine correct Exchange via /symbol (Fixes 100378)
  2. Test DelivType x FundType in phases (Fixes 1010004/100031)
  3. Blacklist 4001005 (parameter conversion errors)
  4. Ensure execution during market hours (unless --force)

Usage:
  python test_spot_minimal.py [--force]
"""

import sys
import time
import yaml
import argparse
from datetime import datetime
from core.auth import KabuAuth
from core.api_client import KabuClient

# --- Config ---
CONFIG_PATH = "config/live_config.yaml"

# Combinations to try (Recommended first, then variants)
PHASE_1 = [(2, None, "Deliv=2(Cash) Fund=None(Omit)")] # First priority
PHASE_2 = [
    (2, "02", "Deliv=2(Cash) Fund='02'(Protect)"),
    (2, "AA", "Deliv=2(Cash) Fund='AA'(Sub)"),
    (0, "02", "Deliv=0(Auto) Fund='02'(Protect)"),
]
PHASE_3 = [
    (2, "  ", "Deliv=2(Cash) Fund='  '(Two spaces)"),
    (0, "  ", "Deliv=0(Auto) Fund='  '(Two spaces)"),
]

DELAY_SEC = 2.0
BLACKLIST = set()

def is_market_open():
    """Simple check for TSE market hours (9:00-11:30, 12:30-15:30)"""
    now = datetime.now()
    if now.weekday() >= 5: return False
    hm = now.hour * 100 + now.minute
    if (900 <= hm <= 1130) or (1230 <= hm <= 1530):
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Run outside market hours")
    args = parser.parse_args()

    print("=" * 60)
    print("  Minimal Spot Buy Diagnosis: 4063 (Shin-Etsu Chemical)")
    print("=" * 60)

    # 1. Check market hours
    if not is_market_open() and not args.force:
        print("  ⚠️ Market is CLOSED. Run with --force if needed.")
        print("  (Orders might fail with market mismatch or status errors)")
        sys.exit(0)

    # 2. Setup client
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        auth = KabuAuth(cfg["api"]["base_url"], cfg["api"]["password"])
        client = KabuClient(cfg["api"]["base_url"], auth)
    except Exception as e:
        print(f"  ❌ Config/Auth error: {e}")
        sys.exit(1)

    # 3. Determine Exchange (Auto-fix 100378)
    print("\n  [Step A] Determining Market...")
    exchange = client.find_exchange("4063")
    symbol_info = client.get_symbol("4063", exchange)
    if not symbol_info:
        print("  ❌ Failed to fetch symbol info. Is API server running?")
        sys.exit(1)
    
    print(f"  ✅ Market: {symbol_info.get('DisplayName')} (Exchange={exchange})")
    print(f"  ✅ Price: {client.get_board('4063', exchange).get('CurrentPrice'):,.0f} JPY")

    # 4. Try combinations (Step B)
    all_phases = [PHASE_1, PHASE_2, PHASE_3]
    print("\n  [Step B] Testing Delivery/Fund Types...")

    for phase_idx, phase in enumerate(all_phases, 1):
        print(f"\n  --- Phase {phase_idx} ---")
        for deliv, fund, desc in phase:
            combo_key = f"{deliv}_{fund}"
            if combo_key in BLACKLIST:
                print(f"  ⏭️  Skipping blacklisted: {desc}")
                continue

            print(f"  ⏳ Testing: {desc}...")
            # Re-auth/Token check implicitly handled in client._post_order via auth
            
            res = client.send_spot_order(
                symbol="4063",
                exchange=exchange,
                side="BUY",
                qty=100,
                order_type=1, # Market
                deliv_type=deliv,
                fund_type=fund
            )

            if res["ok"]:
                print(f"  🎯 SUCCESS! OrderID: {res['order_id']}")
                print(f"\n  Final Result:")
                print(f"    DelivType: {deliv}")
                print(f"    FundType : {fund if fund else '(Omitted)'}")
                print(f"\n  ⚠️ REAL ORDER PLACED. Cancel via kabu STATION if needed.")
                return
            
            # Diagnostic Classification (Step D)
            code = res["code"]
            msg = res["message"]
            
            if code == 4001005:
                print(f"  ❌ Param Error (4001005). Adding to BLACKLIST.")
                BLACKLIST.add(combo_key)
            elif code in [1010004, 100031]:
                print(f"  ❌ Deposit Category Error ({code}): {msg}")
            elif code == 100378:
                print(f"  ❌ Market Mismatch (100378) - Exchange {exchange} rejected.")
            elif code == 4001006:
                print(f"  ⚠️ Rate Limit. Waiting 10s...")
                time.sleep(10)
            else:
                print(f"  ❌ Failed: {code} {msg}")

            time.sleep(DELAY_SEC)

    print("\n" + "=" * 60)
    print("  Diagnosis complete. No successful combination found.")
    print("  Check cash balance and account status.")
    print("=" * 60)

if __name__ == "__main__":
    main()
