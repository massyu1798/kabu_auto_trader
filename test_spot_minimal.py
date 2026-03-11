"""
Minimal Spot (Cash) Buy Diagnosis Tool (v17.0)

PURPOSE
-------
Diagnose spot order parameters (Exchange / DelivType / FundType) by sending
one test order at a time.  Each case gets its own fresh KabuAuth + KabuClient
so that a 401 / 4001009 (API-key mismatch) on one case never contaminates the
result of the next case.

PROBLEM WITH THE OLD APPROACH (v16.x)
---------------------------------------
v16 created a single auth/client at startup and reused it for every case.
After the first order attempt the token sometimes became invalid, causing every
subsequent attempt to return 401 / 4001009 (APIキー不一致).  This made the
results incomparable — only the FIRST case result was trustworthy.

NEW DESIGN (v17.0)
------------------
* run_single_case() creates a brand-new KabuAuth and KabuClient for each run.
* 401 errors are detected and labelled "AUTH_FAIL" — clearly separated from
  order-parameter errors such as 100031 / 4001005.
* CLI supports:
    - Single case:  --exchange 27 --deliv-type 2 --fund-type 02
    - Predefined cases:  --run-default-cases
    - Case by name:  --case recommended
    - Override symbol / qty at will with --symbol / --qty
    - --force to bypass market-hours guard

USAGE
-----
  # Single case (fully specified)
  python test_spot_minimal.py --symbol 4063 --exchange 27 --deliv-type 2 --fund-type 02

  # Single predefined case by name
  python test_spot_minimal.py --case recommended --symbol 4063

  # Run all predefined cases sequentially (fresh auth per case)
  python test_spot_minimal.py --run-default-cases --symbol 4063

  # Force outside market hours
  python test_spot_minimal.py --case recommended --force

ERROR CATEGORIES
----------------
  AUTH       : 401 HTTP / code 4001009  — token/key mismatch
  delivtype  : 1010004 / 100031         — 預り区分エラー
  param_conv : 4001005                  — parameter conversion error
  rate_limit : 4001006                  — rate limited
  mkt_mismatch: 100378                  — Exchange rejected
  unknown    : anything else
"""

import sys
import time
import yaml
import argparse
from datetime import datetime
from typing import Optional

from core.auth import KabuAuth
from core.api_client import KabuClient, classify_order_error

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = "config/live_config.yaml"
DELAY_SEC = 2.0          # seconds to wait between cases (rate-limit guard)

# ---------------------------------------------------------------------------
# Predefined diagnostic cases
# Each dict: exchange, deliv_type, fund_type, label
# ---------------------------------------------------------------------------
DEFAULT_CASES: list[dict] = [
    {
        "label": "recommended",
        "exchange": 27,
        "deliv_type": 2,
        "fund_type": "02",
        "desc": "Exchange=27(TSE+) Deliv=2(Cash) Fund='02'(Protect) [RECOMMENDED]",
    },
    {
        "label": "fund_aa",
        "exchange": 27,
        "deliv_type": 2,
        "fund_type": "AA",
        "desc": "Exchange=27(TSE+) Deliv=2(Cash) Fund='AA'(MarginSub)",
    },
    {
        "label": "fund_01",
        "exchange": 27,
        "deliv_type": 2,
        "fund_type": "01",
        "desc": "Exchange=27(TSE+) Deliv=2(Cash) Fund='01'(General)",
    },
    {
        "label": "deliv3_recommended",
        "exchange": 27,
        "deliv_type": 3,
        "fund_type": "02",
        "desc": "Exchange=27(TSE+) Deliv=3(auMC) Fund='02'(Protect)",
    },
    {
        "label": "deliv3_fund_aa",
        "exchange": 27,
        "deliv_type": 3,
        "fund_type": "AA",
        "desc": "Exchange=27(TSE+) Deliv=3(auMC) Fund='AA'(MarginSub)",
    },
    {
        "label": "ex1_recommended",
        "exchange": 1,
        "deliv_type": 2,
        "fund_type": "02",
        "desc": "Exchange=1(TSE) Deliv=2(Cash) Fund='02'(Protect) [maintenance fallback]",
    },
]

# Friendly names for well-known API codes
_CODE_LABELS: dict[int, tuple[str, str]] = {
    4001009: ("AUTH",         "APIキー不一致 (token mismatch)"),
    4001005: ("param_conv",   "パラメータ変換エラー (bad field value)"),
    4001006: ("rate_limit",   "レート制限 (too many requests)"),
    100031:  ("delivtype",    "預り区分エラー (DelivType/FundType mismatch)"),
    1010004: ("delivtype",    "預り区分エラー (DelivType not allowed)"),
    100378:  ("mkt_mismatch", "Exchange rejected (wrong exchange code)"),
    100368:  ("blocked",      "注文不可 (trading blocked)"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_market_open() -> bool:
    """Simple check for TSE market hours (9:00-11:30, 12:30-15:30 JST)."""
    now = datetime.now()
    if now.weekday() >= 5:          # Saturday / Sunday
        return False
    hm = now.hour * 100 + now.minute
    return (900 <= hm <= 1130) or (1230 <= hm <= 1530)


def load_config() -> dict:
    """Load YAML config. Raises SystemExit on error."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"  ❌ {CONFIG_PATH} not found. Copy from live_config.example.yaml first.")
        sys.exit(1)
    except Exception as e:
        print(f"  ❌ Config load error: {e}")
        sys.exit(1)


def make_auth_and_client(cfg: dict) -> tuple[KabuAuth, KabuClient]:
    """Create a FRESH KabuAuth and KabuClient from config.

    Called once per case so each case starts with a clean token.
    """
    auth = KabuAuth(cfg["api"]["base_url"], cfg["api"]["password"])
    # Eagerly fetch token so we detect auth failures before the order call.
    token = auth.refresh_token()
    if not token:
        raise RuntimeError("Token acquisition failed — is kabu STATION running?")
    client = KabuClient(cfg["api"]["base_url"], auth)
    return auth, client


def classify_result(res: dict) -> tuple[str, str]:
    """Return (category_tag, human_message) from an order result dict."""
    http = res.get("http", 0)
    code = res.get("code", 0)
    msg  = res.get("message", "")

    # HTTP 401 is always an auth failure regardless of api code
    if http == 401:
        label, hint = _CODE_LABELS.get(code, ("AUTH", "認証エラー"))
        return "AUTH_FAIL", f"HTTP 401 / Code={code} [{label}] {hint}"

    if code in _CODE_LABELS:
        label, hint = _CODE_LABELS[code]
        return label, f"Code={code} [{label}] {hint} — {msg}"

    cat = classify_order_error(code)
    return cat, f"Code={code} [{cat}] {msg}"


def print_case_header(idx: Optional[int], total: Optional[int],
                      label: str, desc: str) -> None:
    prefix = f"[{idx}/{total}]" if idx is not None else "[single]"
    print(f"\n{'='*62}")
    print(f"  {prefix} {label}")
    print(f"  {desc}")
    print(f"{'='*62}")


def print_diagnostic(symbol: str, case: dict, res: dict,
                     category: str, category_msg: str,
                     valid: bool) -> None:
    """Print structured per-case diagnostic block."""
    print(f"  ── Diagnostic ─────────────────────────────────────────")
    print(f"     Symbol      : {symbol}")
    print(f"     Exchange    : {case['exchange']}")
    print(f"     DelivType   : {case['deliv_type']}")
    print(f"     FundType    : {case['fund_type']}")
    print(f"     AccountType : 4 (特定口座 — fixed)")
    print(f"     HTTP status : {res.get('http', 0)}")
    print(f"     API code    : {res.get('code', 0)}")
    print(f"     Category    : {category}")
    print(f"     Message     : {res.get('message', '')}")
    if valid:
        print(f"     Validity    : ✅ ORDER-SPEC FAILURE — parameters were received")
        print(f"                   (auth succeeded; result reflects actual order rules)")
    else:
        print(f"     Validity    : ❌ AUTH FAILURE — result is INVALID for comparison")
        print(f"                   (token rejected; order spec NOT tested)")
    print(f"  ────────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Core: run a single case with its own fresh auth/client
# ---------------------------------------------------------------------------

def run_single_case(
    cfg: dict,
    symbol: str,
    exchange: int,
    deliv_type: int,
    fund_type: str,
    qty: int,
    desc: str = "",
    label: str = "custom",
    idx: Optional[int] = None,
    total: Optional[int] = None,
) -> dict:
    """Run one order test case with a FRESH auth/client.

    Returns a summary dict:
        {
          "label": str,
          "ok": bool,           # True = order placed successfully
          "valid": bool,        # True = result is diagnostically meaningful
                                # False = 401 auth failure — ignore result
          "category": str,
          "category_msg": str,
          "res": dict,          # raw result from KabuClient
        }
    """
    print_case_header(idx, total, label, desc or f"Exchange={exchange} DelivType={deliv_type} FundType={fund_type!r}")

    # 1. Fresh auth + client per case
    try:
        _auth, client = make_auth_and_client(cfg)
        print(f"  ✅ Auth OK — fresh token acquired")
    except RuntimeError as e:
        print(f"  ❌ Auth failed: {e}")
        res = {"ok": False, "http": 0, "code": 0,
               "message": str(e), "category": "auth"}
        print_diagnostic(symbol, {"exchange": exchange, "deliv_type": deliv_type,
                                   "fund_type": fund_type},
                         res, "AUTH_FAIL", str(e), valid=False)
        return {"label": label, "ok": False, "valid": False,
                "category": "AUTH_FAIL", "category_msg": str(e), "res": res}

    # 2. Send order
    print(f"  ⏳ Sending order: Symbol={symbol} Exchange={exchange} "
          f"DelivType={deliv_type} FundType={fund_type!r} Qty={qty} ...")
    res = client.send_spot_order(
        symbol=symbol,
        exchange=exchange,
        side="BUY",
        qty=qty,
        order_type=1,       # Market order
        deliv_type=deliv_type,
        fund_type=fund_type,
    )

    # 3. Classify result
    if res["ok"]:
        order_id = res.get("order_id", "?")
        print(f"\n  🎯 SUCCESS! OrderID: {order_id}")
        print(f"  ⚠️  REAL ORDER PLACED — cancel via kabu STATION if needed.")
        print(f"  💡 Working config for live_config.yaml:")
        print(f"     spot:")
        print(f"       exchange: {exchange}")
        print(f"       deliv_type: {deliv_type}")
        print(f'       fund_type: "{fund_type}"')
        return {"label": label, "ok": True, "valid": True,
                "category": "success", "category_msg": f"OrderID={order_id}", "res": res}

    category, category_msg = classify_result(res)
    valid = (category != "AUTH_FAIL")

    # Human-readable per-code guidance
    code = res.get("code", 0)
    http = res.get("http", 0)
    if not valid:
        print(f"  ❌ AUTH FAILURE (HTTP={http} Code={code}): {res.get('message', '')}")
        print(f"     → This result is INVALID. Fresh token was obtained but rejected.")
        print(f"     → Possible cause: kabu STATION restarted or password changed.")
    elif code == 100031:
        print(f"  ❌ 預り区分エラー (100031): {res.get('message', '')}")
        print(f"     → DelivType={deliv_type} / FundType={fund_type!r} not accepted by this account.")
    elif code == 1010004:
        print(f"  ❌ 預り区分エラー (1010004): DelivType={deliv_type} not allowed.")
    elif code == 4001005:
        print(f"  ❌ パラメータ変換エラー (4001005): invalid field value.")
        print(f"     → FundType={fund_type!r} or DelivType={deliv_type} is not a valid code.")
    elif code == 100378:
        print(f"  ❌ Exchange rejected (100378): Exchange={exchange} not available.")
        if exchange == 1:
            print(f"     → Exchange=1 (東証) is only for SOR/東証+ maintenance windows.")
            print(f"     → Use Exchange=27 (東証+) during normal hours.")
    elif code == 4001006:
        print(f"  ⚠️ Rate limit (4001006). Consider increasing DELAY_SEC.")
    else:
        print(f"  ❌ Failed: {category_msg}")

    print_diagnostic(
        symbol=symbol,
        case={"exchange": exchange, "deliv_type": deliv_type, "fund_type": fund_type},
        res=res,
        category=category,
        category_msg=category_msg,
        valid=valid,
    )
    return {"label": label, "ok": False, "valid": valid,
            "category": category, "category_msg": category_msg, "res": res}


# ---------------------------------------------------------------------------
# Market-hours guard (shared)
# ---------------------------------------------------------------------------

def check_market_hours(force: bool) -> None:
    if not is_market_open():
        if not force:
            print("  ⚠️  Market is CLOSED. Run with --force to proceed anyway.")
            print("       (Orders will fail with market/status errors — not auth errors)")
            sys.exit(0)
        print("  ⚠️  Market is CLOSED but --force specified. Proceeding...")


# ---------------------------------------------------------------------------
# Preflight: symbol info (informational only, uses a separate client)
# ---------------------------------------------------------------------------

def show_symbol_info(cfg: dict, symbol: str, qty: int) -> None:
    """Print symbol/price/wallet info. Uses its own ephemeral client."""
    print(f"\n  [Preflight] Fetching symbol info for {symbol} ...")
    try:
        _auth, client = make_auth_and_client(cfg)
    except RuntimeError as e:
        print(f"  ⚠️  Preflight auth failed: {e}")
        return

    symbol_info = client.get_symbol(symbol, 1)
    if not symbol_info:
        print("  ⚠️  Could not fetch symbol info (is kabu STATION running?)")
        return

    name = symbol_info.get("DisplayName", "?")
    unit = symbol_info.get("TradingUnit", "?")
    print(f"  ✅ Name   : {name}")
    print(f"  ✅ Unit   : {unit}")

    board = client.get_board(symbol, 1)
    if board and board.get("CurrentPrice"):
        price = board["CurrentPrice"]
        print(f"  ✅ Price  : {price:,.0f} JPY")
        print(f"  💰 Est.   : {price * qty:,.0f} JPY ({qty} shares)")
    else:
        print("  ⚠️  Could not fetch current price (market may be closed)")

    wallet = client.get_cash_wallet()
    if wallet:
        free_cash = wallet.get("StockAccountWallet", 0)
        print(f"  💳 Cash   : {free_cash:,.0f} JPY (available)")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    print(f"\n{'='*62}")
    print(f"  SUMMARY  ({len(results)} case(s))")
    print(f"{'='*62}")
    for r in results:
        ok_icon    = "🎯" if r["ok"] else ("❌" if r["valid"] else "⚠️")
        valid_note = "" if r["valid"] else " [AUTH FAIL — result invalid]"
        print(f"  {ok_icon} {r['label']:<22} {r['category']:<14} {r['category_msg'][:60]}{valid_note}")
    print(f"{'='*62}")

    auth_failures = [r for r in results if not r["valid"]]
    spec_failures = [r for r in results if r["valid"] and not r["ok"]]
    successes     = [r for r in results if r["ok"]]

    if successes:
        print(f"\n  🎯 {len(successes)} case(s) SUCCEEDED.")
    if auth_failures:
        print(f"\n  ⚠️  {len(auth_failures)} case(s) had AUTH failures (results are INVALID).")
        print(f"      Check that kabu STATION is running and the password is correct.")
    if spec_failures:
        codes = [str(r["res"].get("code", "?")) for r in spec_failures]
        print(f"\n  ❌ {len(spec_failures)} case(s) failed on order spec: codes = {codes}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spot Buy Diagnosis — one fresh auth per case (v17.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_spot_minimal.py --symbol 4063 --exchange 27 --deliv-type 2 --fund-type 02
  python test_spot_minimal.py --case recommended --symbol 4063
  python test_spot_minimal.py --run-default-cases --symbol 4063
  python test_spot_minimal.py --run-default-cases --force
        """,
    )

    # --- Single-case options ---
    parser.add_argument("--exchange",   type=int,   default=None,
                        help="Exchange code (e.g. 27, 1)")
    parser.add_argument("--deliv-type", type=int,   dest="deliv_type", default=None,
                        help="DelivType (e.g. 2, 3)")
    parser.add_argument("--fund-type",  type=str,   dest="fund_type",  default=None,
                        help="FundType (e.g. 02, AA, 01)")
    parser.add_argument("--case",       type=str,   default=None,
                        help=f"Predefined case label: {[c['label'] for c in DEFAULT_CASES]}")

    # --- Batch mode ---
    parser.add_argument("--run-default-cases", action="store_true",
                        help="Run all predefined cases sequentially (fresh auth each)")

    # --- Common options ---
    parser.add_argument("--symbol", default="4063",
                        help="Symbol code to test (default: 4063 Shin-Etsu)")
    parser.add_argument("--qty",    type=int, default=100,
                        help="Order quantity (default: 100)")
    parser.add_argument("--force",  action="store_true",
                        help="Run even when market is closed")

    args = parser.parse_args()

    print("=" * 62)
    print(f"  Spot Buy Diagnosis v17.0 — 1 auth per case")
    print(f"  Symbol: {args.symbol}  Qty: {args.qty}")
    print("=" * 62)

    # Market-hours check
    check_market_hours(args.force)

    # Load config once (all cases share the same credentials)
    cfg = load_config()

    # Show symbol info (uses its own ephemeral auth)
    show_symbol_info(cfg, args.symbol, args.qty)

    results: list[dict] = []

    # ── Mode A: single predefined case by name ──
    if args.case:
        matched = [c for c in DEFAULT_CASES if c["label"] == args.case]
        if not matched:
            print(f"  ❌ Unknown case label: {args.case!r}")
            print(f"     Available: {[c['label'] for c in DEFAULT_CASES]}")
            sys.exit(1)
        c = matched[0]
        r = run_single_case(
            cfg=cfg, symbol=args.symbol,
            exchange=c["exchange"], deliv_type=c["deliv_type"], fund_type=c["fund_type"],
            qty=args.qty, desc=c["desc"], label=c["label"],
        )
        results.append(r)

    # ── Mode B: explicit single case via CLI flags ──
    elif args.exchange is not None or args.deliv_type is not None or args.fund_type is not None:
        exchange   = args.exchange   if args.exchange   is not None else 27
        deliv_type = args.deliv_type if args.deliv_type is not None else 2
        fund_type  = args.fund_type  if args.fund_type  is not None else "02"
        r = run_single_case(
            cfg=cfg, symbol=args.symbol,
            exchange=exchange, deliv_type=deliv_type, fund_type=fund_type,
            qty=args.qty,
            label="custom",
            desc=f"Exchange={exchange} DelivType={deliv_type} FundType={fund_type!r} (manual)",
        )
        results.append(r)

    # ── Mode C: run all default cases ──
    elif args.run_default_cases:
        total = len(DEFAULT_CASES)
        for idx, c in enumerate(DEFAULT_CASES, start=1):
            r = run_single_case(
                cfg=cfg, symbol=args.symbol,
                exchange=c["exchange"], deliv_type=c["deliv_type"], fund_type=c["fund_type"],
                qty=args.qty, desc=c["desc"], label=c["label"],
                idx=idx, total=total,
            )
            results.append(r)
            if r["ok"]:
                print(f"\n  🎯 Working combination found — stopping.")
                break
            if idx < total:
                print(f"  ⏱️  Waiting {DELAY_SEC}s before next case ...")
                time.sleep(DELAY_SEC)

    # ── No mode specified → show help ──
    else:
        parser.print_help()
        print("\n  ℹ️  Tip: run --case recommended to start with the most likely combo.")
        sys.exit(0)

    # Summary
    print_summary(results)


if __name__ == "__main__":
    main()
