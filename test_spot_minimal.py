"""
Minimal Spot (Cash) Buy Diagnosis Tool (v16.1)

Purpose:
  1. Determine correct Exchange via /symbol — prioritize 27(TSE+) for orders
  2. Test DelivType x FundType in phases (Fixes 1010004/100031)
  3. Blacklist 4001005 (parameter conversion errors)
  4. Ensure execution during market hours (unless --force)
  5. Structured diagnostic output per failure

KEY FINDING (from official kabusapi spec + Issue #990):
  - During normal trading hours, new orders REQUIRE Exchange=27 (東証+)
  - Exchange=1 (東証) is ONLY available when SOR/東証+ is in maintenance
  - FundType is REQUIRED for spot orders (omitting it causes 4001005)
  - FundType='02' + DelivType=2 + AccountType=4 is the working combo

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

# Exchange candidates for sendorder (27=TSE+ is primary for orders)
# NOTE: /symbol resolves with Exchange=1 (info), but /sendorder needs 27 (TSE+)
ORDER_EXCHANGE_CANDIDATES = [27, 1]

# Phase 1: Most likely to succeed (based on Issue #990 resolution)
#   AccountType=4(特定), DelivType=2(預り金), FundType='02'(保護預り)
#   Exchange=27(東証+) is tested first
PHASE_1 = [
    (2, "02", "Deliv=2(Cash) Fund='02'(Protect) [RECOMMENDED]"),
]

# Phase 2: Alternative FundType values
PHASE_2 = [
    (2, "AA", "Deliv=2(Cash) Fund='AA'(MarginSub)"),
    (2, "01", "Deliv=2(Cash) Fund='01'(General)"),
]

# Phase 3: DelivType=3 (auマネーコネクト) + FundType variants
PHASE_3 = [
    (3, "02", "Deliv=3(auMC) Fund='02'(Protect)"),
    (3, "AA", "Deliv=3(auMC) Fund='AA'(MarginSub)"),
]

# NEVER include:
# - FundType omitted (causes 4001005 - it's required for spot)
# - FundType='  ' or '' (causes 4001005 - whitespace/empty not valid)
# - FundType='11' (causes 4001005 - not a valid code)
# - DelivType=0 (causes 4001005 - not valid for spot)

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


def try_order(client, symbol, exchange, deliv, fund, desc, qty, market_name):
    """Attempt a single order combo. Returns result dict or None if blacklisted."""
    combo_key = f"{exchange}_{deliv}_{fund}"
    if combo_key in BLACKLIST:
        print(f"  ⏭️  SKIP (blacklisted): {desc} Exchange={exchange}")
        return None

    print(f"\n  ⏳ Testing: {desc} | Exchange={exchange}")

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
        print(f"     FundType  : {fund}")
        print(f"\n  ⚠️ REAL ORDER PLACED. Cancel via kabu STATION if needed.")
        print(f"\n  💡 Add to config/live_config.yaml:")
        print(f"     spot:")
        print(f"       deliv_type: {deliv}")
        print(f"       fund_type: \"{fund}\"")
        return res

    # Failure handling
    code = res.get("code", 0)
    msg = res.get("message", "")

    if code == 4001005:
        print(f"  ❌ BLACKLISTED: Param conversion error (4001005)")
        BLACKLIST.add(combo_key)
    elif code in (1010004, 100031):
        print(f"  ❌ Deposit category error ({code}): {msg}")
    elif code == 100378:
        print(f"  ❌ Market mismatch (100378): Exchange={exchange} rejected")
        if exchange == 1:
            print(f"     💡 NOTE: Exchange=1 (東証) is only for SOR/東証+ maintenance.")
            print(f"     💡 Try Exchange=27 (東証+) for normal trading hours.")
    elif code == 4001006:
        print(f"  ⚠️ Rate limit (4001006). Waiting 10s...")
        time.sleep(10)
    elif code == 100368:
        print(f"  ❌ Order blocked (100368): {msg}")
    else:
        print(f"  ❌ Failed: Code={code} [{res.get('category', '?')}] {msg}")

    log_failure(symbol, exchange, market_name, deliv, fund, res)
    time.sleep(DELAY_SEC)
    return res


def main():
    parser = argparse.ArgumentParser(
        description="Minimal Spot Buy Diagnosis — Exchange=27(TSE+) + DelivType/FundType")
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

    # ── Step A: Get symbol info (using Exchange=1 for /symbol) ──
    print(f"\n  [Step A] Fetching symbol info for {symbol}...")
    symbol_info = client.get_symbol(symbol, 1)
    if not symbol_info:
        print("  ❌ Failed to fetch symbol info. Is kabu STATION running?")
        sys.exit(1)

    market_name = symbol_info.get("DisplayName", "?")
    trading_unit = symbol_info.get("TradingUnit", "?")
    print(f"  ✅ Name   : {market_name}")
    print(f"  ✅ Unit   : {trading_unit}")

    board = client.get_board(symbol, 1)
    if board and board.get("CurrentPrice"):
        price = board["CurrentPrice"]
        print(f"  ✅ Price  : {price:,.0f} JPY")
        print(f"  💰 Est.   : {price * qty:,.0f} JPY ({qty} shares)")
    else:
        print("  ⚠️ Could not fetch current price (market may be closed)")

    wallet = client.get_cash_wallet()
    if wallet:
        free_cash = wallet.get("StockAccountWallet", 0)
        print(f"  💳 Cash   : {free_cash:,.0f} JPY (available)")

    # ── Step B: Test order with Exchange=27 (TSE+) first, then 1 (TSE) ──
    print(f"\n  [Step B] Testing order combinations...")
    print(f"  NOTE: Exchange=27 (東証+) is required for normal-hours orders.")
    print(f"        Exchange=1 (東証) is fallback (SOR/東証+ maintenance only).")

    all_phases = [
        ("Phase 1: Recommended (DelivType=2, FundType='02')", PHASE_1),
        ("Phase 2: Alternative FundType", PHASE_2),
        ("Phase 3: DelivType=3 (auMC)", PHASE_3),
    ]

    for exchange in ORDER_EXCHANGE_CANDIDATES:
        ex_label = "東証+" if exchange == 27 else "東証"
        print(f"\n  {'='*56}")
        print(f"  Exchange={exchange} ({ex_label})")
        print(f"  {'='*56}")

        for phase_name, phase_combos in all_phases:
            print(f"\n  ─── {phase_name} ───")
            for deliv, fund, desc in phase_combos:
                res = try_order(client, symbol, exchange, deliv, fund,
                                desc, qty, f"{market_name}@{ex_label}")
                if res and res["ok"]:
                    return  # Success — exit
                # If 100378 on exchange=27, don't bother with more combos on 27
                if res and res.get("code") == 100378 and exchange == 27:
                    print(f"  ⚠️ Exchange=27 rejected. Falling back to Exchange=1...")
                    break
            else:
                continue
            break  # Break outer phase loop too

    # ── All combinations exhausted ──
    print("\n" + "=" * 60)
    print("  ❌ Diagnosis complete. No successful combination found.")
    print("")
    print("  Key findings from official API spec:")
    print("    - Exchange=27 (東証+) is REQUIRED for normal trading hours")
    print("    - Exchange=1  (東証)  is ONLY for SOR/東証+ maintenance")
    print("    - FundType='02' + DelivType=2 + AccountType=4 is standard")
    print("")
    print("  Checklist:")
    print("    1. Is the market open? (Run with --force to test outside hours)")
    print("    2. Sufficient cash balance? (Check 💳 output above)")
    print("    3. Is kabu STATION properly configured for spot trading?")
    print("    4. Check kabu STATION logs: C:\\Users\\<user>\\AppData\\Local\\kabuSTATION")
    if BLACKLIST:
        print(f"\n  Blacklisted combos (4001005): {BLACKLIST}")
    print("=" * 60)


if __name__ == "__main__":
    main()
