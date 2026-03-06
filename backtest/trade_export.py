"""
Trade export & daily PnL report for backtest results.

Responsibilities:
- Convert AM/PM Trade lists to a unified DataFrame (detail rows)
- Build daily PnL summary (day x session pivot)
- Export CSV / JSON
- Print latest-day summary to console
"""

import pandas as pd
from typing import Any

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ============================================================
# 1. Trade -> row conversion
# ============================================================

def _safe_ts(val) -> str:
    """Convert datetime-like to ISO string safely."""
    if val is None:
        return ""
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    try:
        return pd.Timestamp(val).isoformat()
    except Exception:
        return str(val)


def _safe_date(val):
    """Extract date from datetime-like, return as datetime.date."""
    if val is None:
        return None
    if hasattr(val, "date"):
        return val.date()
    try:
        return pd.Timestamp(val).date()
    except Exception:
        return None


def _side_str(side) -> str:
    """Convert Side enum (or string) to readable string."""
    if hasattr(side, "value"):
        return str(side.value)
    return str(side)


def trades_to_rows(trades: list, session: str) -> list[dict]:
    """
    Convert a list of Trade objects to a list of dicts.
    Uses getattr to absorb field-name differences between AM/PM Trade classes.
    """
    rows = []
    for t in trades:
        entry_dt = getattr(t, "entry_date", None)
        exit_dt = getattr(t, "exit_date", None)
        row = {
            "session": session,
            "ticker": getattr(t, "ticker", ""),
            "side": _side_str(getattr(t, "side", "")),
            "entry_dt": _safe_ts(entry_dt),
            "exit_dt": _safe_ts(exit_dt),
            "day": _safe_date(entry_dt),
            "entry_price": getattr(t, "entry_price", 0),
            "exit_price": getattr(t, "exit_price", 0),
            "size": getattr(t, "size", 0),
            "pnl": getattr(t, "pnl", 0),
            "pnl_pct": getattr(t, "pnl_pct", None),
            "entry_reason": getattr(t, "entry_reason", ""),
            "exit_reason": getattr(t, "exit_reason", ""),
        }
        # pnl_pct may not exist in all Trade classes
        if row["pnl_pct"] is None:
            ep = row["entry_price"]
            sz = row["size"]
            if ep and sz:
                row["pnl_pct"] = row["pnl"] / (ep * sz) * 100
            else:
                row["pnl_pct"] = 0.0
        rows.append(row)
    return rows


# ============================================================
# 2. Build unified DataFrame
# ============================================================

def build_trades_df(morning_trades: list, afternoon_trades: list) -> pd.DataFrame:
    """
    Merge AM and PM trades into a single DataFrame.
    Returns DataFrame sorted by entry_dt.
    """
    am_rows = trades_to_rows(morning_trades, "AM")
    pm_rows = trades_to_rows(afternoon_trades, "PM")
    all_rows = am_rows + pm_rows
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df.sort_values("entry_dt", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================
# 3. Daily PnL summary
# ============================================================

def build_daily_pnl(df: pd.DataFrame, last_days: int | None = None) -> pd.DataFrame:
    """
    Build day x session PnL pivot table.
    Columns: day, AM_pnl, PM_pnl, TOTAL, AM_cnt, PM_cnt, TOTAL_cnt
    Filtered to last N business days (by unique trade days in data).
    """
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(["day", "session"]).agg(
        pnl=("pnl", "sum"),
        cnt=("pnl", "count"),
    ).reset_index()

    pivot_pnl = grouped.pivot_table(
        index="day", columns="session", values="pnl", fill_value=0
    )
    pivot_cnt = grouped.pivot_table(
        index="day", columns="session", values="cnt", fill_value=0
    )

    # Ensure AM/PM columns exist
    for col in ["AM", "PM"]:
        if col not in pivot_pnl.columns:
            pivot_pnl[col] = 0
        if col not in pivot_cnt.columns:
            pivot_cnt[col] = 0

    daily = pd.DataFrame({
        "day": pivot_pnl.index,
        "AM_pnl": pivot_pnl["AM"].values,
        "PM_pnl": pivot_pnl["PM"].values,
        "TOTAL": (pivot_pnl["AM"] + pivot_pnl["PM"]).values,
        "AM_cnt": pivot_cnt["AM"].astype(int).values,
        "PM_cnt": pivot_cnt["PM"].astype(int).values,
        "TOTAL_cnt": (pivot_cnt["AM"] + pivot_cnt["PM"]).astype(int).values,
    })
    daily.sort_values("day", inplace=True)
    daily.reset_index(drop=True, inplace=True)

    if last_days is not None and last_days > 0:
        unique_days = sorted(daily["day"].unique())
        target_days = unique_days[-last_days:]
        daily = daily[daily["day"].isin(target_days)].reset_index(drop=True)

    return daily


# ============================================================
# 4. Latest day summary
# ============================================================

def get_latest_day_summary(df: pd.DataFrame) -> dict:
    """
    Get summary for the latest day in trade data.
    Returns dict with keys: day, AM_pnl, PM_pnl, TOTAL, AM_cnt, PM_cnt, TOTAL_cnt
    """
    if df.empty:
        return {}

    latest = df["day"].max()
    latest_df = df[df["day"] == latest]

    am = latest_df[latest_df["session"] == "AM"]
    pm = latest_df[latest_df["session"] == "PM"]

    return {
        "day": latest,
        "AM_pnl": am["pnl"].sum() if not am.empty else 0,
        "PM_pnl": pm["pnl"].sum() if not pm.empty else 0,
        "TOTAL": am["pnl"].sum() + pm["pnl"].sum(),
        "AM_cnt": len(am),
        "PM_cnt": len(pm),
        "TOTAL_cnt": len(latest_df),
    }


# ============================================================
# 5. Export utilities
# ============================================================

def export_trades_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  -> Trade detail exported: {path} ({len(df)} rows)")


def export_trades_json(df: pd.DataFrame, path: str) -> None:
    df.to_json(path, orient="records", force_ascii=False, indent=2,
               date_format="iso")
    print(f"  -> Trade detail exported: {path} ({len(df)} rows)")


def export_daily_csv(daily_df: pd.DataFrame, path: str) -> None:
    daily_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  -> Daily PnL exported: {path} ({len(daily_df)} days)")


def export_daily_json(daily_df: pd.DataFrame, path: str) -> None:
    # Convert date to string for JSON
    out = daily_df.copy()
    out["day"] = out["day"].astype(str)
    out.to_json(path, orient="records", force_ascii=False, indent=2)
    print(f"  -> Daily PnL exported: {path} ({len(daily_df)} days)")


# ============================================================
# 6. Console print utilities
# ============================================================

def print_daily_table(daily_df: pd.DataFrame) -> None:
    """Print daily PnL table to console."""
    if daily_df.empty:
        print("  (no trade data)")
        return

    # Format for display
    display = daily_df.copy()
    display["day"] = display["day"].astype(str)
    display["AM_pnl"] = display["AM_pnl"].map(lambda x: f"{x:+,.0f}")
    display["PM_pnl"] = display["PM_pnl"].map(lambda x: f"{x:+,.0f}")
    display["TOTAL"] = display["TOTAL"].map(lambda x: f"{x:+,.0f}")

    if HAS_TABULATE:
        print(tabulate(
            display[["day", "AM_pnl", "AM_cnt", "PM_pnl", "PM_cnt", "TOTAL", "TOTAL_cnt"]].values.tolist(),
            headers=["Day", "AM PnL", "AM#", "PM PnL", "PM#", "TOTAL", "#"],
            tablefmt="simple",
            colalign=("left", "right", "right", "right", "right", "right", "right"),
        ))
    else:
        # Fallback: simple print
        header = f"  {'Day':<12} {'AM PnL':>12} {'AM#':>4} {'PM PnL':>12} {'PM#':>4} {'TOTAL':>12} {'#':>4}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, row in display.iterrows():
            print(f"  {row['day']:<12} {row['AM_pnl']:>12} {row['AM_cnt']:>4} "
                  f"{row['PM_pnl']:>12} {row['PM_cnt']:>4} {row['TOTAL']:>12} {row['TOTAL_cnt']:>4}")


def print_latest_day(summary: dict) -> None:
    """Print latest day summary to console."""
    if not summary:
        print("  (no trade data for latest day)")
        return

    day = summary["day"]
    print(f"\n■ Latest Day in Data: {day}")
    print(f"  [AM] PnL: {summary['AM_pnl']:>+14,.0f} 円  ({summary['AM_cnt']} trades)")
    print(f"  [PM] PnL: {summary['PM_pnl']:>+14,.0f} 円  ({summary['PM_cnt']} trades)")
    print(f"  TOTAL:    {summary['TOTAL']:>+14,.0f} 円  ({summary['TOTAL_cnt']} trades)")
