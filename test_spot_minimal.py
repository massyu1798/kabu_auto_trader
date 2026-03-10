"""
Minimal Spot (Cash) Buy Diagnosis Tool (v16.0)

Purpose:
  1. Determine correct Exchange via /symbol (Fixes 100378)
  2. Test DelivType x FundType in phases (Fixes 1010004/100031)
  3. Blacklist 4001005 (parameter conversion errors)
  4. Ensure execution during market hours (unless --force)
  5. Structured diagnostic output per failure

Usage:
  python test_spot_minimal.py [--force] [--symbol 4063] [--qty 100]
"""

import sys
import time
import yaml
import argparse
from datetime import datetime
from core.auth import KabuAuth
from core.api_client import KabuClient, classify_order_error

# --- Config ---
CONFIG_PATH = "config/live_config.yaml"

# Phase 1: Recommended default (most likely to succeed for spot)
PHASE_1 = [
    (2, None, "Deliv=2(Cash) Fund=omitted"),
]

# Phase 2: Explicit FundType variants
PHASE_2 = [
    (2, "02", "Deliv=2(Cash) Fund='02'(Protect)"),
    (2, "AA", "Deliv=2(Cash) Fund='AA'(MarginSub)"),
    (2, "11", "Deliv=2(Cash) Fund='11'(Deposit)"),
]

# Phase 3: DelivType=0 (auto) variants — last resort
PHASE_3 = [
    (0, None, "Deliv=0(Auto) Fund=omitted"),
    (0, "02", "Deliv=0(Auto) Fund='02'(Protect)"),
]

# Do NOT include whitespace-only FundType — it causes 4001005

DELAY_SEC = 2.0
BLACKLIST = set()  # combo keys that returned 4001005


def is_market_open() -> bool:
    """Simple check for TSE market hours (9:00-11:30, 12:30-15:30 JST)."""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    hm = now.hour * 100 + now.minute
    return (900 <= hm <= 1130) or (1230 <= hm <= 1530)


def log_failure(symbol: str, exchange: int, market_name: str,
                deliv: int, fund, res: dict) -> None:
    """Print structured diagnostic log for a failed order attempt."""
    code = res.get("code", 0)
    category = res.get("category", classify_order_error(code))
    msg = res.get("message", "")
    http = res.get("http", 0)

    print(f"     ── Diagnostic ──")
    print(f"     Symbol    : {symbol}")
    print(f"     Exchange  : {exchange} ({market_name})")
    print(f"     DelivType : {deliv}")
    print(f"     FundType  : {fund if fund else '(omitted)'}")
    print(f"     HTTP      : {http}")
    print(f"     API Code  : {code}")
    print(f"     Category  : {category}")
    print(f"     Message   : {msg}")


def main():
    parser = argparse.ArgumentParser(
        description="Minimal Spot Buy Diagnosis — Exchange auto-detect + DelivType/FundType phased test")
    parser.add_argument("--force", action="store_true",
                        help="Run outside market hours")
    parser.add_argument("--symbol", default="4063",
                        help="Symbol to test (default: 4063 Shin-Etsu)")
    parser.add_argument("--qty", type=int, default=100,
                        help="Order quantity (default: 100)")
    args = parser.parse_args()

    symbol = args.symbol
    qty = args.qty

    print("=" * 60)
    print(f"  Minimal Spot Buy Diagnosis: {symbol}")
    print("=" * 60)

    # ── Step 0: Check market hours ──
    if not is_market_open():
        if not args.force:
            print("  ⚠️ Market is CLOSED. Run with --force to proceed anyway.")
            print("  (Orders will likely fail with market/status errors)")
            sys.exit(0)
        else:
            print("  ⚠️ Market is CLOSED but --force specified. Proceeding...")

    # ── Step 1: Setup client ──
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        auth = KabuAuth(cfg["api"]["base_url"], cfg["api"]["password"])
        client = KabuClient(cfg["api"]["base_url"], auth)
    except FileNotFoundError:
        print(f"  ❌ {CONFIG_PATH} not found. Copy from live_config.example.yaml first.")
        sys.exit(1)
    except Exception as e:
        print(f"  ❌ Config/Auth error: {e}")
        sys.exit(1)

    # ── Step A: Determine Exchange (auto-fix 100378) ──
    print(f"\n  [Step A] Determining Market for {symbol}...")
    exchange = client.find_exchange(symbol)
    symbol_info = client.get_symbol(symbol, exchange)
    if not symbol_info:
        print("  ❌ Failed to fetch symbol info. Is kabu STATION running?")
        sys.exit(1)

    market_name = symbol_info.get("DisplayName", "?")
    trading_unit = symbol_info.get("TradingUnit", "?")
    print(f"  ✅ Market : {market_name} (Exchange={exchange})")
    print(f"  ✅ Unit   : {trading_unit}")

    board = client.get_board(symbol, exchange)
    if board and board.get("CurrentPrice"):
        price = board["CurrentPrice"]
        print(f"  ✅ Price  : {price:,.0f} JPY")
        print(f"  💰 Est.   : {price * qty:,.0f} JPY ({qty} shares)")
    else:
        print("  ⚠️ Could not fetch current price (market may be closed)")

    # Check cash wallet balance
    wallet = client.get_cash_wallet()
    if wallet:
        free_cash = wallet.get("StockAccountWallet", 0)
        print(f"  💳 Cash   : {free_cash:,.0f} JPY (available)")

    # ── Step B: Test DelivType x FundType in phases ──
    all_phases = [
        ("Phase 1 (Recommended)", PHASE_1),
        ("Phase 2 (Explicit FundType)", PHASE_2),
        ("Phase 3 (DelivType=0 fallback)", PHASE_3),
    ]
    print(f"\n  [Step B] Testing Delivery/Fund Type combinations...")

    for phase_name, phase_combos in all_phases:
        print(f"\n  ─── {phase_name} ───")
        for deliv, fund, desc in phase_combos:
            combo_key = f"{deliv}_{fund}"
            if combo_key in BLACKLIST:
                print(f"  ⏭️  SKIP (blacklisted): {desc}")
                continue

            print(f"\n  ⏳ Testing: {desc}")

            res = client.send_spot_order(
                symbol=symbol,
                exchange=exchange,
                side="BUY",
                qty=qty,
                order_type=1,  # Market order
                deliv_type=deliv,
                fund_type=fund,
            )

            if res["ok"]:
                print(f"\n  🎯 SUCCESS! OrderID: {res['order_id']}")
                print(f"\n  ── Working Configuration ──")
                print(f"     Exchange  : {exchange} ({market_name})")
                print(f"     DelivType : {deliv}")
                print(f"     FundType  : {fund if fund else '(omitted)'}")
                print(f"\n  ⚠️ REAL ORDER PLACED. Cancel via kabu STATION if needed.")
                print(f"\n  💡 Add to config/live_config.yaml:")
                print(f"     spot:")
                print(f"       deliv_type: {deliv}")
                if fund:
                    print(f"       fund_type: \"{fund}\"")
                else:
                    print(f"       # fund_type: omitted (API default)")
                return

            # ── Failure handling ──
            code = res.get("code", 0)
            category = res.get("category", classify_order_error(code))
            msg = res.get("message", "")

            if code == 4001005:
                print(f"  ❌ BLACKLISTED: Param conversion error (4001005)")
                BLACKLIST.add(combo_key)
            elif code in (1010004, 100031):
                print(f"  ❌ Deposit category error ({code}): {msg}")
            elif code == 100378:
                print(f"  ❌ Market mismatch (100378): Exchange={exchange} rejected")
                print(f"     This should not happen with auto-detect. Check API status.")
            elif code == 4001006:
                print(f"  ⚠️ Rate limit (4001006). Waiting 10s...")
                time.sleep(10)
            elif code == 100368:
                print(f"  ❌ Order blocked (100368): {msg}")
                print(f"     sendorder API may be entirely blocked. Aborting.")
                log_failure(symbol, exchange, market_name, deliv, fund, res)
                sys.exit(1)
            else:
                print(f"  ❌ Failed: Code={code} [{category}] {msg}")

            log_failure(symbol, exchange, market_name, deliv, fund, res)
            time.sleep(DELAY_SEC)

    # ── All combinations exhausted ──
    print("\n" + "=" * 60)
    print("  ❌ Diagnosis complete. No successful combination found.")
    print("")
    print("  Checklist:")
    print("    1. Is the market open? (Run with --force to test outside hours)")
    print("    2. Sufficient cash balance? (Check 💳 output above)")
    print("    3. Is kabu STATION account configured for spot trading?")
    print("    4. Check kabu STATION logs for additional error details.")
    if BLACKLIST:
        print(f"\n  Blacklisted combos (4001005): {BLACKLIST}")
    print("=" * 60)


if __name__ == "__main__":
    main()
