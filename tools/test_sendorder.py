#!/usr/bin/env python3
"""
Single-shot /sendorder test tool for MarginTradeType diagnosis.

Usage:
  python tools/test_sendorder.py --symbol 4063 --side BUY --qty 100 --trade_type 3
  python tools/test_sendorder.py --symbol 4063 --side BUY --qty 100 --trade_type 1
  python tools/test_sendorder.py --symbol 4063 --side BUY --qty 100 --all_types

Options:
  --symbol       Symbol code (e.g. 4063)
  --side         BUY or SELL
  --qty          Quantity (e.g. 100)
  --trade_type   MarginTradeType: 1=制度信用, 2=一般信用(長期), 3=一般信用(デイトレ)
  --all_types    Test all MarginTradeType values (1,2,3) sequentially
  --dry_run      Build payload but do NOT send (default: False)
  --exchange     Exchange code (default: 1 = 東証)

This script sends a REAL order unless --dry_run is specified.
Use with caution during market hours.

MarginTradeType reference (kabu STATION API):
  1 = 制度信用
  2 = 一般信用（長期）
  3 = 一般信用（デイトレ）
"""

import argparse
import json
import sys
import os
import time
import yaml
import requests

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.auth import KabuAuth


TRADE_TYPE_LABELS = {
    1: "制度信用",
    2: "一般信用(長期)",
    3: "一般信用(デイトレ)",
}


def mask_password(payload: dict) -> dict:
    """Return a copy with Password masked."""
    masked = dict(payload)
    if "Password" in masked:
        masked["Password"] = "********"
    return masked


def build_payload(auth: KabuAuth, symbol: str, exchange: int,
                  side: str, qty: int, margin_trade_type: int) -> dict:
    """Build /sendorder payload for margin new order."""
    side_code = "2" if side.upper() == "BUY" else "1"
    return {
        "Password": auth.api_password,
        "Symbol": symbol,
        "Exchange": exchange,
        "SecurityType": 1,
        "Side": side_code,
        "CashMargin": 2,              # 2=新規
        "MarginTradeType": margin_trade_type,
        "DelivType": 0,
        "FundType": "  ",
        "AccountType": 4,             # 4=特定
        "Qty": qty,
        "FrontOrderType": 10,         # 10=成行
        "Price": 0,
        "ExpireDay": 0,               # 0=当日
    }


def send_order(base_url: str, auth: KabuAuth, payload: dict) -> dict:
    """Send /sendorder and return structured result."""
    headers = auth.get_headers()
    if not headers:
        return {"ok": False, "http": 0, "code": 0, "message": "no auth token"}

    try:
        res = requests.post(
            f"{base_url}/sendorder",
            headers=headers,
            json=payload,
            timeout=10,
        )
        if res.status_code == 200:
            body = res.json()
            return {"ok": True, "http": 200, "code": 0,
                    "order_id": body.get("OrderId", ""), "raw": body}

        api_code = 0
        api_msg = ""
        try:
            err_body = res.json()
            api_code = err_body.get("Code", 0)
            api_msg = err_body.get("Message", res.text[:300])
        except Exception:
            api_msg = res.text[:300]

        return {"ok": False, "http": res.status_code,
                "code": api_code, "message": api_msg}

    except requests.exceptions.Timeout:
        return {"ok": False, "http": 0, "code": 0, "message": "timeout"}
    except Exception as e:
        return {"ok": False, "http": 0, "code": 0, "message": str(e)}


def test_single(base_url: str, auth: KabuAuth, symbol: str, exchange: int,
                side: str, qty: int, trade_type: int, dry_run: bool):
    """Test a single MarginTradeType."""
    label = TRADE_TYPE_LABELS.get(trade_type, f"unknown({trade_type})")
    print(f"\n{'='*60}")
    print(f"  MarginTradeType={trade_type} ({label})")
    print(f"  Symbol={symbol} Side={side} Qty={qty} Exchange={exchange}")
    print(f"{'='*60}")

    payload = build_payload(auth, symbol, exchange, side, qty, trade_type)

    # Log masked payload
    print(f"\n  Payload (password masked):")
    masked = mask_password(payload)
    for k, v in masked.items():
        print(f"    {k}: {v}")

    if dry_run:
        print(f"\n  [DRY RUN] Skipping actual /sendorder call")
        return None

    print(f"\n  Sending /sendorder ...")
    result = send_order(base_url, auth, payload)

    if result["ok"]:
        print(f"  ✅ SUCCESS  http={result['http']} OrderId={result.get('order_id','')}")
        print(f"     ⚠️ REAL ORDER PLACED - cancel manually if needed!")
    else:
        print(f"  ❌ FAILED   http={result['http']} Code={result['code']}")
        print(f"     Message: {result['message']}")

        if result["code"] == 100368:
            print(f"     → 信用新規抑止: この MarginTradeType({trade_type}={label}) では注文できません")
        elif result["code"] == 4003001:
            print(f"     → ワンショット金額エラー: 数量を減らすか余力を確認してください")
        elif result["code"] == 4001006:
            print(f"     → API実行回数エラー: しばらく待ってからリトライしてください")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test /sendorder with different MarginTradeType values")
    parser.add_argument("--symbol", required=True, help="Symbol code (e.g. 4063)")
    parser.add_argument("--side", required=True, choices=["BUY", "SELL"], help="BUY or SELL")
    parser.add_argument("--qty", type=int, default=100, help="Quantity (default: 100)")
    parser.add_argument("--trade_type", type=int, choices=[1, 2, 3],
                        help="MarginTradeType: 1=制度, 2=一般(長期), 3=一般(デイトレ)")
    parser.add_argument("--all_types", action="store_true",
                        help="Test all MarginTradeType values (1, 2, 3)")
    parser.add_argument("--exchange", type=int, default=1, help="Exchange (default: 1=東証)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Build payload but do NOT send")
    parser.add_argument("--config", default="config/live_config.yaml",
                        help="Path to live_config.yaml")
    args = parser.parse_args()

    if not args.all_types and args.trade_type is None:
        parser.error("Either --trade_type or --all_types is required")

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_url = cfg["api"]["base_url"]
    api_password = cfg["api"]["password"]
    timeout = cfg.get("api", {}).get("timeout_sec", 10)

    print(f"Config: {args.config}")
    print(f"API base: {base_url}")
    print(f"Symbol: {args.symbol}  Side: {args.side}  Qty: {args.qty}")

    if args.dry_run:
        print("*** DRY RUN MODE - no real orders will be placed ***")
    else:
        print("*** LIVE MODE - real orders WILL be placed! ***")
        print("    Press Ctrl+C within 3 seconds to cancel...")
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            print("\nCancelled.")
            return

    # Auth
    auth = KabuAuth(base_url, api_password, timeout=timeout)
    token = auth.get_token()
    if not token:
        print("❌ Failed to get auth token. Is kabu STATION running?")
        return

    # Test
    if args.all_types:
        types_to_test = [1, 2, 3]
    else:
        types_to_test = [args.trade_type]

    results = {}
    for tt in types_to_test:
        result = test_single(base_url, auth, args.symbol, args.exchange,
                             args.side, args.qty, tt, args.dry_run)
        results[tt] = result
        if len(types_to_test) > 1 and tt != types_to_test[-1]:
            print("\n  Waiting 1.5s before next test...")
            time.sleep(1.5)

    # Summary
    if len(types_to_test) > 1 and not args.dry_run:
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        for tt, r in results.items():
            label = TRADE_TYPE_LABELS.get(tt, "?")
            if r is None:
                status = "SKIPPED"
            elif r["ok"]:
                status = f"✅ SUCCESS (OrderId={r.get('order_id','')})"
            else:
                status = f"❌ FAILED Code={r['code']} {r['message'][:60]}"
            print(f"  MarginTradeType={tt} ({label}): {status}")

        print(f"\n  Recommendation:")
        success_types = [tt for tt, r in results.items() if r and r["ok"]]
        if 3 in success_types:
            print(f"  → Use margin_trade_type: 3 (デイトレ) in live_config.yaml")
        elif 1 in success_types:
            print(f"  → Use margin_trade_type: 1 (制度信用) in live_config.yaml")
        elif 2 in success_types:
            print(f"  → Use margin_trade_type: 2 (一般信用長期) in live_config.yaml")
        else:
            print(f"  → All types failed. Check symbol restrictions or account settings.")


if __name__ == "__main__":
    main()
