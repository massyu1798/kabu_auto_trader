#!/usr/bin/env python3
"""
現物 vs 信用 切り分け診断ツール (v16.0)

100368「現在、株式信用新規の注文は抑止されております。」が
「信用新規だけ」なのか「sendorder 全般」なのかを一発で切り分けます。

Usage:
  # 現物買いだけテスト
  python tools/test_sendorder_cash.py --symbol 4063 --side BUY --qty 100

  # 現物 + 信用(制度/一般/デイトレ)を全部テスト
  python tools/test_sendorder_cash.py --symbol 4063 --side BUY --qty 100 --all

  # ドライラン（payload確認のみ、実際の発注なし）
  python tools/test_sendorder_cash.py --symbol 4063 --side BUY --qty 100 --all --dry_run

テスト対象:
  CashMargin=1  現物（MarginTradeType なし）
  CashMargin=2  信用新規 MarginTradeType=1（制度信用）
  CashMargin=2  信用新規 MarginTradeType=2（一般信用 長期）
  CashMargin=2  信用新規 MarginTradeType=3（一般信用 デイトレ）

判定:
  現物 ✅ / 信用 ❌ → 信用新規だけが抑止（銘柄 or 口座の信用規制）
  現物 ❌ / 信用 ❌ → sendorder 全般が抑止（API障害 or 銘柄売買停止）
  現物 ❌ / 信用 ✅ → 現物だけ抑止（余力不足 等）

⚠️ --dry_run を付けないと本当に発注されます。注意！
"""

import argparse
import sys
import os
import time
import yaml
import requests

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.auth import KabuAuth


ORDER_TYPES = {
    "cash":       {"label": "現物",                  "CashMargin": 1, "MarginTradeType": None},
    "margin_1":   {"label": "信用新規(制度信用)",    "CashMargin": 2, "MarginTradeType": 1},
    "margin_2":   {"label": "信用新規(一般 長期)",   "CashMargin": 2, "MarginTradeType": 2},
    "margin_3":   {"label": "信用新規(一般 デイトレ)","CashMargin": 2, "MarginTradeType": 3},
}


def mask_password(payload: dict) -> dict:
    """Return a copy with Password masked."""
    masked = dict(payload)
    if "Password" in masked:
        masked["Password"] = "********"
    return masked


def build_payload(auth: KabuAuth, symbol: str, exchange: int,
                  side: str, qty: int,
                  cash_margin: int, margin_trade_type: int | None) -> dict:
    """Build /sendorder payload for cash or margin order.

    Key fix (v16.0): No empty-string/whitespace FundType.
    - Cash (CashMargin=1): DelivType=2, FundType omitted (API default)
    - Margin (CashMargin=2): DelivType=0, FundType omitted
    """
    side_code = "2" if side.upper() == "BUY" else "1"

    payload = {
        "Password": auth.api_password,
        "Symbol": symbol,
        "Exchange": exchange,
        "SecurityType": 1,
        "Side": side_code,
        "CashMargin": cash_margin,
        "DelivType": 2 if cash_margin == 1 else 0,
        "AccountType": 4,             # 4=特定
        "Qty": qty,
        "FrontOrderType": 10,         # 10=成行
        "Price": 0,
        "ExpireDay": 0,               # 0=当日
    }
    # Note: FundType intentionally omitted (was "  " which caused 4001005)

    # MarginTradeType is only for margin orders (CashMargin=2,3)
    if margin_trade_type is not None:
        payload["MarginTradeType"] = margin_trade_type

    return payload


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


def resolve_exchange(base_url: str, auth: KabuAuth, symbol: str) -> int:
    """Auto-detect the correct Exchange for a symbol via /symbol endpoint."""
    headers = auth.get_headers()
    if not headers:
        return 1
    candidates = [1, 3, 5, 6]
    for ex in candidates:
        try:
            res = requests.get(
                f"{base_url}/symbol/{symbol}@{ex}",
                headers=headers,
                timeout=10,
            )
            if res.status_code == 200:
                data = res.json()
                if data.get("Symbol") == symbol:
                    resolved = data.get("Exchange", ex)
                    name = data.get("DisplayName", "?")
                    print(f"  ✅ Exchange resolved: {symbol} -> {resolved} ({name})")
                    return resolved
        except Exception:
            continue
    print(f"  ⚠️ Could not resolve Exchange for {symbol}. Using default 1.")
    return 1


def test_single(base_url: str, auth: KabuAuth, symbol: str, exchange: int,
                side: str, qty: int, order_key: str, dry_run: bool) -> dict | None:
    """Test a single order type."""
    ot = ORDER_TYPES[order_key]
    label = ot["label"]
    cm = ot["CashMargin"]
    mtt = ot["MarginTradeType"]

    print(f"\n{'='*60}")
    print(f"  [{order_key}] {label}")
    print(f"  CashMargin={cm}" + (f"  MarginTradeType={mtt}" if mtt else "  (現物 - MTT なし)"))
    print(f"  Symbol={symbol} Side={side} Qty={qty} Exchange={exchange}")
    print(f"{'='*60}")

    payload = build_payload(auth, symbol, exchange, side, qty, cm, mtt)

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
            print(f"     → 信用新規抑止: このタイプ({label}) では注文できません")
        elif result["code"] == 4003001:
            print(f"     → ワンショット金額エラー")
        elif result["code"] == 4001006:
            print(f"     → API実行回数エラー: しばらく待ってください")
        elif result["code"] == 4001005:
            print(f"     → パラメータ変換エラー: payloadの型/値を確認")
        elif result["code"] == 100378:
            print(f"     → 市場コードエラー: Exchange={exchange} がこの銘柄に合わない可能性")
        elif result["code"] in (1010004, 100031):
            print(f"     → 預り区分エラー: DelivType/FundType の組み合わせが不正")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="現物 vs 信用 切り分け診断 — 100368 が信用だけかを判定 (v16.0)")
    parser.add_argument("--symbol", required=True, help="Symbol code (e.g. 4063)")
    parser.add_argument("--side", required=True, choices=["BUY", "SELL"], help="BUY or SELL")
    parser.add_argument("--qty", type=int, default=100, help="Quantity (default: 100)")
    parser.add_argument("--all", action="store_true",
                        help="現物 + 信用(制度/一般/デイトレ)の4パターン全て試す")
    parser.add_argument("--exchange", type=int, default=None,
                        help="Exchange (default: auto-detect via /symbol)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Build payload but do NOT send")
    parser.add_argument("--config", default="config/live_config.yaml",
                        help="Path to live_config.yaml")
    args = parser.parse_args()

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
        print("\n*** DRY RUN MODE - no real orders will be placed ***")
    else:
        print("\n*** LIVE MODE - real orders WILL be placed! ***")
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

    # Auto-detect Exchange if not specified
    if args.exchange is not None:
        exchange = args.exchange
        print(f"\n  Using specified Exchange: {exchange}")
    else:
        print(f"\n  Auto-detecting Exchange for {args.symbol}...")
        exchange = resolve_exchange(base_url, auth, args.symbol)

    # Determine which tests to run
    if args.all:
        keys_to_test = ["cash", "margin_1", "margin_2", "margin_3"]
    else:
        keys_to_test = ["cash"]  # default: 現物のみ

    results = {}
    for key in keys_to_test:
        result = test_single(base_url, auth, args.symbol, exchange,
                             args.side, args.qty, key, args.dry_run)
        results[key] = result
        if key != keys_to_test[-1]:
            print("\n  Waiting 1.5s before next test...")
            time.sleep(1.5)

    # Summary
    if not args.dry_run:
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")

        for key, r in results.items():
            ot = ORDER_TYPES[key]
            label = ot["label"]
            if r is None:
                status = "SKIPPED"
            elif r["ok"]:
                status = f"✅ SUCCESS (OrderId={r.get('order_id','')})"
            else:
                status = f"❌ FAILED Code={r['code']} {r['message'][:60]}"
            print(f"  {label:30s}: {status}")

        # Diagnosis
        cash_ok = results.get("cash") and results["cash"]["ok"]

        if len(keys_to_test) > 1:
            margin_any_ok = any(
                results.get(k) and results[k]["ok"]
                for k in ["margin_1", "margin_2", "margin_3"]
                if k in results
            )
            margin_all_fail = all(
                results.get(k) and not results[k]["ok"]
                for k in ["margin_1", "margin_2", "margin_3"]
                if k in results
            )

            print(f"\n  ── 診断 ──")
            if cash_ok and margin_all_fail:
                print(f"  → 現物 ✅ / 信用 全❌")
                print(f"  → 信用新規だけが抑止されています（銘柄規制 or 口座設定）")
                print(f"  → kabuステーション or 証券会社の信用新規規制を確認してください")
            elif cash_ok and margin_any_ok:
                print(f"  → 現物 ✅ / 信用 一部✅")
                print(f"  → 特定の信用区分だけ抑止。通った区分を live_config.yaml に設定してください")
            elif not cash_ok and margin_all_fail:
                print(f"  → 現物 ❌ / 信用 全❌")
                print(f"  → sendorder 全般が失敗（API障害 or 銘柄売買停止）")
            elif not cash_ok and margin_any_ok:
                print(f"  → 現物 ❌ / 信用 一部✅")
                print(f"  → 現物だけ失敗（余力不足 etc.）")
        else:
            print(f"\n  ── 診断 ──")
            if cash_ok:
                print(f"  → 現物 ✅ → sendorder API 自体は正常")
                print(f"  → 100368 は信用新規だけの抑止とほぼ確定")
                print(f"  → --all で信用タイプ別の切り分けも実行できます")
            else:
                code = results["cash"]["code"] if results.get("cash") else "?"
                print(f"  → 現物 ❌ (Code={code})")
                print(f"  → sendorder 全般が失敗の可能性。--all で信用も確認してください")


if __name__ == "__main__":
    main()
