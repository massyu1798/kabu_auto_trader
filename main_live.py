"""
JP Stock Auto Trading Bot v18.0 - multi-lot hold_entries + spec-aligned margin
- Morning (9:05-11:00): EnsembleEngine (BUY + SELL) + daily bias filter
- Afternoon (config entry_start-entry_end): AfternoonReversalEngine (BUY + SELL)
- 5-min OHLCV bars from /board API
- Screener-based watchlist (same as BT)
- Short (SELL) position support: symmetric SL/TP/trailing
- Live risk caps: max_positions, max_notional_per_position, max_total_exposure
- Configurable margin_trade_type (1=制度, 2=一般, 3=デイトレ)
- Total exposure check (BT-aligned) with safety margin
- Force close: PM at force_close_time, all at market close
- Cooldown: config-based bars (BT-aligned)
- API error handling: blacklist, order throttle, break on 429
- v18.0: Exchange=27 for new orders (issue #1072),
         DelivType=0(新規)/2(返済), FundType='11',
         multi-lot hold_entries from /positions, safe close flow
"""

import copy
import math
import time
import yaml
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, date as date_type, timedelta
from core.auth import KabuAuth
from core.api_client import KabuClient
from core.order_manager import OrderManager, LivePosition, HoldEntry, MARGIN_TRADE_TYPE_LABELS
from core.bar_builder import BarBuilder
from strategy.ensemble import EnsembleEngine
from strategy.afternoon_reversal import AfternoonReversalEngine
from strategy.overnight_gap import generate_ong_signals, BLACKOUT_DATES
from strategy.simple_momentum import SimpleMomentumEngine
from strategy.universe import UNIVERSE
from backtest.screener import screen_stocks

# Minimum completed 5-min bars required for signal calculation
MIN_BARS = 10

VERSION = "v18.0"

# ============================================
# Constants for API error handling / throttle
# ============================================

# Safety margin: effective_cap = floor(raw_cap * SAFETY_MARGIN_RATIO)
SAFETY_MARGIN_RATIO = 0.98

# Max orders per signal check (prevents burst sendorder)
MAX_ORDERS_PER_CHECK_AM = 2
MAX_ORDERS_PER_CHECK_PM = 1

# Interval (sec) between sendorder calls to avoid 429
ORDER_INTERVAL_SEC = 0.8

# API error codes
CODE_ONESHOT_AMOUNT = 4003001   # ワンショット：金額エラー
CODE_RATE_LIMIT = 4001006       # API実行回数エラー
CODE_MARGIN_BLOCKED = 100368    # 信用新規抑止

# Retry shrink ratio for 4003001
RETRY_SHRINK_RATIO = 0.80

# Block duration for 4003001 retry failure (minutes)
BLOCK_MINUTES_ONESHOT = 30

# Block duration for unknown errors (minutes)
BLOCK_MINUTES_UNKNOWN = 10

# Default Exchange for new-open orders (per issue #1072)
# Exchange=1 (東証) is DEPRECATED for new orders since 2026-02.
# Use 27 (東証+) or 9 (SOR).
DEFAULT_ORDER_EXCHANGE = 27

# ONG: MarginTradeType=2 (一般信用長期) — overnight hold required
ONG_MARGIN_TRADE_TYPE = 2
# SM: MarginTradeType=3 (一般信用デイトレ) — same as AM/PM
SM_MARGIN_TRADE_TYPE = 3


def load_config():
    with open("config/live_config.yaml", "r", encoding="utf-8") as f:
        live = yaml.safe_load(f)
    with open("config/strategy_config.yaml", "r", encoding="utf-8") as f:
        strategy = yaml.safe_load(f)
    try:
        with open("config/afternoon_config.yaml", "r", encoding="utf-8") as f:
            afternoon = yaml.safe_load(f)
    except FileNotFoundError:
        afternoon = None
    try:
        with open("config/overnight_config.yaml", "r", encoding="utf-8") as f:
            ong = yaml.safe_load(f)
    except FileNotFoundError:
        ong = None
    try:
        with open("config/simple_momentum_config.yaml", "r", encoding="utf-8") as f:
            sm = yaml.safe_load(f)
    except FileNotFoundError:
        sm = None
    return live, strategy, afternoon, ong, sm


# ============================================
# Screener: BT-aligned watchlist
# ============================================

def run_screener(strat_cfg: dict, fallback_watchlist: list[str]) -> list[str]:
    """
    Run screen_stocks() to generate watchlist (same as BT).
    Returns list of kabuSte ticker codes (e.g. "6902").
    Falls back to config watchlist on failure.
    """
    print("\n■ Running screener (BT-aligned)...")
    try:
        selected = screen_stocks(strat_cfg)
        if not selected:
            print("  ⚠️ Screener returned 0 stocks, using fallback watchlist")
            ticker_map = {t: f"{t}.T" for t in fallback_watchlist}
            return fallback_watchlist, ticker_map

        # Convert "6902.T" -> "6902" for kabuSte API
        tickers = []
        ticker_map = {}  # "6902" -> "6902.T" for yfinance
        for s in selected:
            yf_ticker = s["ticker"]  # e.g. "6902.T"
            kabu_ticker = yf_ticker.replace(".T", "")
            tickers.append(kabu_ticker)
            ticker_map[kabu_ticker] = yf_ticker

        print(f"  -> Screener selected {len(tickers)} stocks")
        for i, s in enumerate(selected[:10]):
            print(f"    {i+1}. {s['ticker']} close={s['close']:.0f} ATR%={s['atr_pct']:.2f} score={s['score']:.2f}")
        if len(selected) > 10:
            print(f"    ... and {len(selected)-10} more")

        return tickers, ticker_map
    except Exception as e:
        print(f"  ❌ Screener failed: {e}")
        print(f"  -> Using fallback watchlist ({len(fallback_watchlist)} stocks)")
        ticker_map = {t: f"{t}.T" for t in fallback_watchlist}
        return fallback_watchlist, ticker_map


# ============================================
# Daily Bias Filter (BT-aligned)
# ============================================

def calc_daily_bias(strat_cfg: dict, yf_tickers: list[str]) -> dict[str, str]:
    """
    Calculate daily bias for each ticker using yfinance daily data.
    Returns dict: yf_ticker -> "BULL" / "BEAR" / "NEUTRAL"
    BT uses daily_bias.ema_short/ema_long (default 5/25).
    """
    dc = strat_cfg.get("daily_bias", {})
    ema_short = dc.get("ema_short", 5)
    ema_long = dc.get("ema_long", 25)

    print(f"\n■ Calculating daily bias (EMA {ema_short}/{ema_long})...")
    bias = {}
    today = date_type.today()

    for yf_ticker in yf_tickers:
        try:
            data = yf.download(yf_ticker, period="60d", interval="1d", progress=False)
            if data is None or data.empty or len(data) < ema_long + 1:
                bias[yf_ticker] = "NEUTRAL"
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0].lower() for col in data.columns]
            else:
                data.columns = [col.lower() for col in data.columns]

            ema_s = ta.ema(data["close"], length=ema_short)
            ema_l = ta.ema(data["close"], length=ema_long)

            if ema_s is None or ema_l is None or pd.isna(ema_s.iloc[-1]) or pd.isna(ema_l.iloc[-1]):
                bias[yf_ticker] = "NEUTRAL"
            elif float(ema_s.iloc[-1]) > float(ema_l.iloc[-1]):
                bias[yf_ticker] = "BULL"
            else:
                bias[yf_ticker] = "BEAR"
        except Exception:
            bias[yf_ticker] = "NEUTRAL"

    bulls = sum(1 for v in bias.values() if v == "BULL")
    bears = sum(1 for v in bias.values() if v == "BEAR")
    neutrals = sum(1 for v in bias.values() if v == "NEUTRAL")
    print(f"  -> BULL={bulls} BEAR={bears} NEUTRAL={neutrals}")
    return bias


# ============================================
# Time window helpers (config-based)
# ============================================

def is_market_open() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.hour * 100 + now.minute
    return 900 <= t <= 1525


def is_morning_entry() -> bool:
    now = datetime.now()
    t = now.hour * 100 + now.minute
    return 905 <= t <= 1100


def is_afternoon_entry(entry_start: int = 1230, entry_end: int = 1400) -> bool:
    now = datetime.now()
    t = now.hour * 100 + now.minute
    return entry_start <= t <= entry_end


def is_force_close_pm(force_close_time: int = 1450) -> bool:
    now = datetime.now()
    t = now.hour * 100 + now.minute
    return t >= force_close_time


def is_market_close_force() -> bool:
    now = datetime.now()
    t = now.hour * 100 + now.minute
    return t >= 1520


def is_ong_exit_window() -> bool:
    """9:00-9:05: ONG翌朝成行決済ウィンドウ"""
    now = datetime.now()
    t = now.hour * 100 + now.minute
    return 900 <= t <= 905


def is_sm_exit_window() -> bool:
    """12:30-12:34: SM後場寄り成行決済ウィンドウ"""
    now = datetime.now()
    t = now.hour * 100 + now.minute
    return 1230 <= t <= 1234


def is_ong_entry_window() -> bool:
    """14:50-15:00: ONG引け前発注ウィンドウ"""
    now = datetime.now()
    t = now.hour * 100 + now.minute
    return 1450 <= t <= 1500


def get_session_label(pm_start: int = 1230, pm_end: int = 1400) -> str:
    if is_morning_entry():
        return "AM"
    if is_afternoon_entry(pm_start, pm_end):
        return "PM"
    return "--"


def calc_atr_from_bars(df: pd.DataFrame, period: int = 14) -> float | None:
    if df is None or len(df) < period + 1:
        return None
    atr_series = ta.atr(df["high"], df["low"], df["close"], length=period)
    if atr_series is None or pd.isna(atr_series.iloc[-1]):
        return None
    return float(atr_series.iloc[-1])


# ============================================
# Live risk caps: per-position + total exposure
# ============================================

def apply_live_risk_caps(
    positions: list,
    new_price: float,
    new_size: int,
    max_notional_per_position: float,
    max_total_exposure: float,
    session: str,
    ticker: str,
    now_str: str,
) -> int | None:
    raw_size = new_size

    max_size_per_pos = int(max_notional_per_position / new_price)
    max_size_per_pos = (max_size_per_pos // 100) * 100

    if max_size_per_pos < 100:
        print(f"  [{now_str}] [{session}] {ticker}: SKIP "
              f"(per-position cap {max_notional_per_position:,.0f} too small for price={new_price:.0f})")
        return None

    if new_size > max_size_per_pos:
        notional_before = new_price * new_size
        new_size = max_size_per_pos
        notional_after = new_price * new_size
        print(f"  [{now_str}] [{session}] {ticker}: size capped {raw_size} -> {new_size} "
              f"(per-position cap {max_notional_per_position:,.0f}, "
              f"notional {notional_before:,.0f} -> {notional_after:,.0f})")

    existing_exposure = sum(p.entry_price * p.size for p in positions)
    remaining = max_total_exposure - existing_exposure

    if remaining <= 0:
        print(f"  [{now_str}] [{session}] {ticker}: SKIP (total exposure full) "
              f"existing={existing_exposure:,.0f} max_total={max_total_exposure:,.0f}")
        return None

    max_size_total = int(remaining / new_price)
    max_size_total = (max_size_total // 100) * 100

    if max_size_total < 100:
        print(f"  [{now_str}] [{session}] {ticker}: SKIP (remaining exposure too small) "
              f"remaining={remaining:,.0f} price={new_price:.0f}")
        return None

    if new_size > max_size_total:
        size_before = new_size
        new_size = max_size_total
        new_notional = new_price * new_size
        print(f"  [{now_str}] [{session}] {ticker}: size capped {size_before} -> {new_size} "
              f"(total exposure cap {max_total_exposure:,.0f}, "
              f"existing={existing_exposure:,.0f} remaining={remaining:,.0f} "
              f"notional={new_notional:,.0f})")

    effective_cap = math.floor(max_total_exposure * SAFETY_MARGIN_RATIO)
    total_after = existing_exposure + new_price * new_size
    if total_after > effective_cap:
        adj_remaining = effective_cap - existing_exposure
        if adj_remaining <= 0:
            print(f"  [{now_str}] [{session}] {ticker}: SKIP (safety margin exceeded) "
                  f"effective_cap={effective_cap:,.0f}")
            return None
        adj_size = int(adj_remaining / new_price)
        adj_size = (adj_size // 100) * 100
        if adj_size < 100:
            print(f"  [{now_str}] [{session}] {ticker}: SKIP (safety margin: remaining too small)")
            return None
        if adj_size < new_size:
            print(f"  [{now_str}] [{session}] {ticker}: size capped {new_size} -> {adj_size} "
                  f"(safety margin {SAFETY_MARGIN_RATIO}, effective_cap={effective_cap:,.0f})")
            new_size = adj_size

    return new_size


# ============================================
# Blacklist management (per-ticker order block)
# ============================================

class TickerBlacklist:
    def __init__(self):
        self._blocked: dict[str, tuple[datetime, str]] = {}

    def block(self, ticker: str, until: datetime, reason: str):
        self._blocked[ticker] = (until, reason)
        print(f"  🚫 BLOCKED {ticker} until {until.strftime('%H:%M:%S')} reason={reason}")

    def block_until_eod(self, ticker: str, reason: str):
        today = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
        self.block(ticker, today, reason)

    def is_blocked(self, ticker: str) -> bool:
        if ticker not in self._blocked:
            return False
        until, reason = self._blocked[ticker]
        if datetime.now() >= until:
            del self._blocked[ticker]
            return False
        return True

    def get_reason(self, ticker: str) -> str:
        if ticker in self._blocked:
            until, reason = self._blocked[ticker]
            remaining = (until - datetime.now()).total_seconds()
            return f"{reason} (unblock in {remaining:.0f}s at {until.strftime('%H:%M:%S')})"
        return ""

    def daily_reset(self):
        self._blocked.clear()


# ============================================
# Priority scoring for order candidates
# ============================================

def calc_priority_score(signal: str, info: dict) -> float:
    rsi = info.get("rsi") or 50.0
    mm = abs(info.get("morning_move") or 0.0)

    if signal == "SELL":
        score = (rsi / 100.0) * 0.6 + min(mm / 5.0, 1.0) * 0.4
    else:
        score = ((100.0 - rsi) / 100.0) * 0.6 + min(mm / 5.0, 1.0) * 0.4
    return score


# ============================================
# Order execution with error handling
# ============================================

def execute_entry_with_error_handling(
    order_mgr: OrderManager,
    ticker: str,
    signal: str,
    entry_price: float,
    size: int,
    stop_loss: float,
    take_profit: float,
    reason: str,
    session: str,
    blacklist: TickerBlacklist,
    now_str: str,
    exchange: int = DEFAULT_ORDER_EXCHANGE,
) -> str:
    """Execute entry with error-code-based handling.

    Args:
        exchange: Exchange for new-open sendorder (default: 27=東証+).
    """

    time.sleep(ORDER_INTERVAL_SEC)

    result = order_mgr.entry(
        ticker, signal, entry_price, size,
        stop_loss, take_profit, reason, session=session,
        exchange=exchange,
    )

    if result.get("ok"):
        return "ok"

    code = result.get("code", 0)
    http_status = result.get("http", 0)

    if http_status == 429 or code == CODE_RATE_LIMIT:
        print(f"  ⛔ [{session}] {ticker}: 429 rate limit (code={code}) "
              f"-> stop further orders this cycle")
        return "break"

    if code == CODE_MARGIN_BLOCKED:
        mtt = order_mgr.margin_trade_type
        mtt_label = MARGIN_TRADE_TYPE_LABELS.get(mtt, "?")
        print(f"  ⛔ [{session}] {ticker}: 100368 margin blocked "
              f"(MTT={mtt}={mtt_label}) -> block until EOD")
        blacklist.block_until_eod(ticker,
                                  f"100368 MTT={mtt}({mtt_label}): {result.get('message','')}")
        return "skip"

    if code == CODE_ONESHOT_AMOUNT:
        retry_size = int(size * RETRY_SHRINK_RATIO)
        retry_size = (retry_size // 100) * 100
        if retry_size < 100:
            print(f"  ⛔ [{session}] {ticker}: 4003001 oneshot -> shrink retry impossible (size too small)")
            blacklist.block(ticker,
                            datetime.now() + timedelta(minutes=BLOCK_MINUTES_ONESHOT),
                            f"4003001 no viable size")
            return "skip"

        retry_notional = entry_price * retry_size
        print(f"  🔄 [{session}] {ticker}: 4003001 oneshot -> shrink retry "
              f"size {size}->{retry_size} notional={retry_notional:,.0f}")

        time.sleep(ORDER_INTERVAL_SEC)

        retry_result = order_mgr.entry(
            ticker, signal, entry_price, retry_size,
            stop_loss, take_profit, reason + " [shrink-retry]", session=session,
            exchange=exchange,
        )

        if retry_result.get("ok"):
            return "retry_ok"

        retry_code = retry_result.get("code", 0)
        retry_http = retry_result.get("http", 0)

        if retry_http == 429 or retry_code == CODE_RATE_LIMIT:
            print(f"  ⛔ [{session}] {ticker}: retry also got 429 -> break cycle")
            return "break"

        print(f"  ⛔ [{session}] {ticker}: shrink retry also failed "
              f"(code={retry_code}) -> block {BLOCK_MINUTES_ONESHOT}min")
        blacklist.block(ticker,
                        datetime.now() + timedelta(minutes=BLOCK_MINUTES_ONESHOT),
                        f"4003001 retry failed (code={retry_code})")
        return "skip"

    print(f"  ⚠️ [{session}] {ticker}: unknown error http={http_status} code={code} "
          f"-> block {BLOCK_MINUTES_UNKNOWN}min")
    blacklist.block(ticker,
                    datetime.now() + timedelta(minutes=BLOCK_MINUTES_UNKNOWN),
                    f"unknown error http={http_status} code={code}")
    return "skip"


# ============================================
# Position Management: BUY/SELL symmetric
# ============================================

def update_trailing_stop(pos: LivePosition, bars_df: pd.DataFrame,
                         trail_mult: float, atr_val: float) -> None:
    if bars_df is None or len(bars_df) < 2:
        return
    current = float(bars_df["close"].iloc[-1])
    if pos.side == "BUY":
        current_high = float(bars_df["high"].iloc[-1])
        if current > pos.entry_price:
            profit_ratio = (current_high - pos.entry_price) / pos.entry_price
            accel = max(0.7, 1.0 - profit_ratio * 2.0)
            new_trail = current_high - (atr_val * trail_mult * accel)
            if new_trail > pos.trailing_stop:
                pos.trailing_stop = new_trail
    else:
        current_low = float(bars_df["low"].iloc[-1])
        if current < pos.entry_price:
            profit_ratio = (pos.entry_price - current_low) / pos.entry_price
            accel = max(0.7, 1.0 - profit_ratio * 2.0)
            new_trail = current_low + (atr_val * trail_mult * accel)
            if new_trail < pos.trailing_stop:
                pos.trailing_stop = new_trail


def check_exit(pos: LivePosition, current: float) -> str | None:
    if pos.side == "BUY":
        if current <= pos.stop_loss:
            return f"SL hit (price={current:.0f} <= SL={pos.stop_loss:.0f})"
        if current >= pos.take_profit:
            return f"TP hit (price={current:.0f} >= TP={pos.take_profit:.0f})"
        if current <= pos.trailing_stop:
            return f"Trailing hit (price={current:.0f} <= trail={pos.trailing_stop:.0f})"
    else:
        if current >= pos.stop_loss:
            return f"SL hit (price={current:.0f} >= SL={pos.stop_loss:.0f})"
        if current <= pos.take_profit:
            return f"TP hit (price={current:.0f} <= TP={pos.take_profit:.0f})"
        if current >= pos.trailing_stop:
            return f"Trailing hit (price={current:.0f} >= trail={pos.trailing_stop:.0f})"
    return None


def calc_entry_params(side: str, entry_price: float, sl_dist: float, tp_dist: float):
    if side == "BUY":
        stop_loss = entry_price - sl_dist
        take_profit = entry_price + tp_dist
        trailing_stop = stop_loss
    else:
        stop_loss = entry_price + sl_dist
        take_profit = entry_price - tp_dist
        trailing_stop = stop_loss
    return stop_loss, take_profit, trailing_stop


# ============================================
# ONG: Load historical daily data for RSI/ATR
# ============================================

def load_ong_daily_data(ong_cfg: dict) -> dict[str, pd.DataFrame]:
    """ONG対象銘柄の日足データをyfinanceから取得（起動時に一度だけ呼ぶ）。

    RSI(2)・ATR(14)計算に必要な過去データを取得する。
    当日データは 14:50 に /board から取得するため、ここでは除外する。
    """
    tickers_yf = ong_cfg.get("tickers", [])
    today_dt = date_type.today()
    daily_data: dict[str, pd.DataFrame] = {}

    print(f"\n■ Loading ONG daily history ({len(tickers_yf)} tickers)...")
    for yf_ticker in tickers_yf:
        kabu_ticker = yf_ticker.replace(".T", "")
        try:
            data = yf.download(yf_ticker, period="60d", interval="1d", progress=False)
            if data is None or data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0].lower() for col in data.columns]
            else:
                data.columns = [col.lower() for col in data.columns]
            # Exclude today's row — will be appended from /board at 14:50
            data = data[data.index.date < today_dt]
            if len(data) < 20:
                print(f"  ⚠️ {yf_ticker}: insufficient history ({len(data)} rows) — skip")
                continue
            daily_data[kabu_ticker] = data
        except Exception as e:
            print(f"  ⚠️ ONG daily load failed for {yf_ticker}: {e}")

    print(f"  -> Loaded history for {len(daily_data)}/{len(tickers_yf)} tickers")
    return daily_data


# ============================================
# Main
# ============================================
def main():
    live_cfg, strat_cfg, afternoon_cfg, ong_cfg, sm_cfg = load_config()

    api_timeout = live_cfg.get("api", {}).get("timeout_sec",
                  live_cfg.get("api", {}).get("timeout", 10))

    auth = KabuAuth(live_cfg["api"]["base_url"], live_cfg["api"]["password"],
                    timeout=api_timeout)
    client = KabuClient(live_cfg["api"]["base_url"], auth, timeout=api_timeout)
    order_mgr = OrderManager(client, live_cfg)

    blacklist = TickerBlacklist()

    # ── ONG order manager (MarginTradeType=2: 一般信用長期) ──────────────
    ong_enabled = live_cfg.get("ong", {}).get("enabled", False) and ong_cfg is not None
    ong_order_mgr: OrderManager | None = None
    ong_tickers_kabu: list[str] = []
    ong_daily_data: dict[str, pd.DataFrame] = {}

    if ong_enabled:
        _ong_order_cfg = copy.deepcopy(live_cfg)
        _ong_order_cfg["trade"]["margin_trade_type"] = ONG_MARGIN_TRADE_TYPE
        _ong_order_cfg["trade"]["max_positions"] = live_cfg.get("ong", {}).get("max_positions", 3)
        _ong_order_cfg["trade"]["max_daily_loss"] = 1.0   # 100% → effectively disables can_entry daily-loss gate for ONG
        _ong_order_cfg["trade"]["initial_capital"] = ong_cfg["global"]["initial_capital"]
        ong_order_mgr = OrderManager(client, _ong_order_cfg)

        ong_tickers_kabu = [t.replace(".T", "") for t in ong_cfg.get("tickers", [])]
        ong_daily_data = load_ong_daily_data(ong_cfg)

    # ── SM order manager (MarginTradeType=3: 一般信用デイトレ) ───────────
    sm_enabled = live_cfg.get("sm", {}).get("enabled", False) and sm_cfg is not None
    sm_order_mgr: OrderManager | None = None
    sm_tickers_kabu: list[str] = []
    sm_engine: SimpleMomentumEngine | None = None
    sm_bar_builder: BarBuilder | None = None

    if sm_enabled:
        _sm_max = live_cfg.get("sm", {}).get("max_positions_per_side", 2) * 2
        _sm_order_cfg = copy.deepcopy(live_cfg)
        _sm_order_cfg["trade"]["margin_trade_type"] = SM_MARGIN_TRADE_TYPE
        _sm_order_cfg["trade"]["max_positions"] = _sm_max
        _sm_order_cfg["trade"]["max_daily_loss"] = sm_cfg["risk"]["max_daily_loss"]
        _sm_order_cfg["trade"]["initial_capital"] = sm_cfg["global"]["initial_capital"]
        sm_order_mgr = OrderManager(client, _sm_order_cfg)

        sm_tickers_kabu = [t.replace(".T", "") for t in UNIVERSE.keys()]
        sm_engine = SimpleMomentumEngine("config/simple_momentum_config.yaml")
        sm_bar_builder = BarBuilder(bar_interval_min=5)
        print(f"  SM: {len(sm_tickers_kabu)} tickers loaded from UNIVERSE")

    trade_cfg = live_cfg.get("trade", {})
    live_max_positions = trade_cfg.get("max_positions", 2)
    live_max_notional_per_position = float(trade_cfg.get("max_notional_per_position", 3_000_000))
    live_max_total_exposure = float(trade_cfg.get("max_total_exposure", 6_000_000))

    live_mtt = order_mgr.margin_trade_type
    live_mtt_label = MARGIN_TRADE_TYPE_LABELS.get(live_mtt, "?")

    use_screener = live_cfg.get("screener", {}).get("enabled", True)
    fallback_watchlist = live_cfg.get("watchlist", [])

    if use_screener:
        watchlist, ticker_map = run_screener(strat_cfg, fallback_watchlist)
    else:
        watchlist = fallback_watchlist
        ticker_map = {t: f"{t}.T" for t in watchlist}
        print(f"\n■ Screener disabled, using config watchlist ({len(watchlist)} stocks)")

    pm_available = afternoon_cfg is not None
    if pm_available:
        ar_cfg = afternoon_cfg["afternoon_reversal"]
        pm_entry_start = ar_cfg.get("entry_start", 1230)
        pm_entry_end = ar_cfg.get("entry_end", 1400)
        pm_force_close = afternoon_cfg["exit"].get("force_close_time", 1450)
    else:
        pm_entry_start = 1230
        pm_entry_end = 1400
        pm_force_close = 1450

    yf_tickers = list(ticker_map.values())
    daily_bias = calc_daily_bias(strat_cfg, yf_tickers)

    am_cd = strat_cfg.get("cooldown", {})
    am_cd_enabled = am_cd.get("enabled", False)
    am_cd_loss_bars = am_cd.get("bars_after_loss", 15)
    am_cd_win_bars = am_cd.get("bars_after_win", 5)
    bar_interval = 5
    am_cd_loss_min = am_cd_loss_bars * bar_interval
    am_cd_win_min = am_cd_win_bars * bar_interval

    pm_cd = afternoon_cfg.get("cooldown", {}) if afternoon_cfg else {}
    pm_cd_enabled = pm_cd.get("enabled", False)
    pm_cd_loss_bars = pm_cd.get("bars_after_loss", 6)
    pm_cd_win_bars = pm_cd.get("bars_after_win", 2)
    pm_cd_loss_min = pm_cd_loss_bars * bar_interval
    pm_cd_win_min = pm_cd_win_bars * bar_interval

    order_mgr.cooldown_config = {
        "am_enabled": am_cd_enabled,
        "am_loss_min": am_cd_loss_min,
        "am_win_min": am_cd_win_min,
        "pm_enabled": pm_cd_enabled,
        "pm_loss_min": pm_cd_loss_min,
        "pm_win_min": pm_cd_win_min,
    }

    bar_builder = BarBuilder(bar_interval_min=5)

    print(f"\nLoading EnsembleEngine (AM)...")
    am_engine = EnsembleEngine("config/strategy_config.yaml")

    pm_engine = None
    if pm_available:
        print(f"Loading AfternoonReversalEngine (PM)...")
        pm_engine = AfternoonReversalEngine("config/afternoon_config.yaml")

    last_signal_time = 0
    signal_check_count = 0
    current_day = None

    # ONG/SM: one-time daily action flags (reset on day change)
    ong_exit_done_today: bool = False
    ong_entry_done_today: bool = False
    sm_signal_done_today: bool = False
    sm_entry_done_today: bool = False
    sm_exit_done_today: bool = False
    sm_long_tickers_pending: list[str] = []
    sm_short_tickers_pending: list[str] = []

    # per-ticker Exchange from board (for /board info queries only)
    ticker_exchange: dict[str, int] = {}

    ensemble_cfg = strat_cfg["ensemble"]
    am_initial_capital = float(strat_cfg["global"]["initial_capital"])
    pm_initial_capital = float(afternoon_cfg["global"]["initial_capital"]) if afternoon_cfg else am_initial_capital

    print(f"\n{'='*60}")
    print(f"Bot running... ({VERSION} multi-lot hold_entries + spec-aligned margin)")
    print(f"{'='*60}")
    print(f"  Watchlist: {len(watchlist)} stocks (screener={'ON' if use_screener else 'OFF'})")
    print(f"  AM Initial capital: {am_initial_capital:,.0f}")
    print(f"  PM Initial capital: {pm_initial_capital:,.0f}")
    print(f"  API timeout: {api_timeout}s")
    print(f"  Safety margin: {SAFETY_MARGIN_RATIO} (effective = raw * {SAFETY_MARGIN_RATIO})")
    print(f"  ── Live Risk Caps ──")
    print(f"    max_positions:              {live_max_positions}")
    print(f"    max_notional_per_position:  {live_max_notional_per_position:,.0f}")
    print(f"    max_total_exposure:         {live_max_total_exposure:,.0f}")
    print(f"    margin_trade_type:          {live_mtt} ({live_mtt_label})")
    print(f"  Max orders/check: AM={MAX_ORDERS_PER_CHECK_AM} PM={MAX_ORDERS_PER_CHECK_PM}")
    print(f"  Order interval: {ORDER_INTERVAL_SEC}s")
    print(f"  ── Order params (v18.0) ──")
    print(f"    Exchange (new-open):    {DEFAULT_ORDER_EXCHANGE} (東証+)")
    print(f"    Exchange (close):       from hold_entries (per-position)")
    print(f"    DelivType (new-open):   0 (指定なし)")
    print(f"    DelivType (close):      2 (お預り金)")
    print(f"    FundType:               11 (信用取引)")
    print(f"    AccountType:            4 (特定口座)")
    print(f"    MarginTradeType:        {live_mtt} ({live_mtt_label})")
    print(f"    ClosePositions.HoldID:  from /positions ExecutionID (multi-lot)")
    print(f"  [AM] Entry: 9:05-11:00 | buy_thr={ensemble_cfg['buy_threshold']} sell_thr={ensemble_cfg['sell_threshold']}")
    print(f"       SL: {strat_cfg['exit']['stop_loss_atr_multiplier']} ATR | TP: {strat_cfg['exit']['take_profit_rr_ratio']} R:R")
    print(f"       Daily bias: EMA {strat_cfg.get('daily_bias',{}).get('ema_short',5)}/{strat_cfg.get('daily_bias',{}).get('ema_long',25)} (BEAR=disabled)")
    print(f"       Cooldown: {'ON' if am_cd_enabled else 'OFF'} loss={am_cd_loss_min}min win={am_cd_win_min}min")
    if pm_available:
        print(f"  [PM] Entry: {pm_entry_start}-{pm_entry_end} | force_close={pm_force_close}")
        ar = afternoon_cfg["afternoon_reversal"]
        pm_exit = afternoon_cfg["exit"]
        print(f"       RSI<={ar['rsi_oversold']}/{ar['rsi_overbought']} + BB + VWAP + morning_move>={ar['min_morning_move_pct']}%")
        print(f"       SL: {pm_exit['stop_loss_atr_multiplier']} ATR | TP: {pm_exit['take_profit_atr_multiplier']} ATR")
        print(f"       Cooldown: {'ON' if pm_cd_enabled else 'OFF'} loss={pm_cd_loss_min}min win={pm_cd_win_min}min")
    else:
        print("  [PM] afternoon_config.yaml not found - PM session disabled")
    if ong_enabled:
        _omtt = MARGIN_TRADE_TYPE_LABELS.get(ONG_MARGIN_TRADE_TYPE, "?")
        print(f"  [ONG] Entry@14:50 Exit@9:00-9:05 | {len(ong_tickers_kabu)} tickers "
              f"MTT={ONG_MARGIN_TRADE_TYPE}({_omtt}) max_pos={ong_order_mgr.trade_config['max_positions']}")
    else:
        print("  [ONG] disabled")
    if sm_enabled:
        print(f"  [SM] Signal@11:25 Entry@11:30 Exit@12:30 | {len(sm_tickers_kabu)} tickers "
              f"risk_per_pos={sm_cfg['global']['risk_per_position']}")
    else:
        print("  [SM] disabled")
    print(f"  Min bars: {MIN_BARS}")
    print(f"  Force close: PM@{pm_force_close} / All@1520")

    # ONG: Recover open ONG positions from /positions at startup
    if ong_order_mgr and ong_tickers_kabu:
        print("\n■ Checking for existing ONG positions to recover...")
        try:
            # product="2" = 信用取引 (margin) per kabu Station API spec
            _raw_positions = client.get_positions(product="2")
            for _pd in _raw_positions:
                _sym = str(_pd.get("Symbol", ""))
                if _sym not in ong_tickers_kabu:
                    continue
                _side_code = str(_pd.get("Side", ""))
                _side = "BUY" if _side_code == "2" else "SELL"
                _qty = int(_pd.get("LeavesQty", 0))
                if _qty <= 0:
                    continue
                _exec_id = str(_pd.get("ExecutionID", ""))
                if not _exec_id:
                    continue
                _price = float(_pd.get("Price", 0))
                _exch = int(_pd.get("Exchange", DEFAULT_ORDER_EXCHANGE))
                # Use ExecutionDay from /positions as entry_time if available,
                # otherwise fall back to current time (position is still valid)
                _exec_day_str = str(_pd.get("ExecutionDay", ""))
                try:
                    _entry_time = datetime.strptime(_exec_day_str, "%Y%m%d%H%M%S")
                except (ValueError, TypeError):
                    _entry_time = datetime.now()
                _lp = LivePosition(
                    ticker=_sym, side=_side,
                    entry_price=_price, entry_time=_entry_time,
                    size=_qty, stop_loss=_price * 0.9, take_profit=_price * 1.1,
                    trailing_stop=_price * 0.9,
                    exchange=_exch, order_exchange=_exch,
                    hold_entries=[HoldEntry(hold_id=_exec_id, qty=_qty,
                                            exchange=_exch, price=_price)],
                    reason="recovered from /positions at startup",
                    session="ONG",
                )
                ong_order_mgr.positions.append(_lp)
                print(f"  ✅ Recovered ONG: {_sym} {_side} × {_qty} @ {_price:.0f} (ExecID={_exec_id})")
            if not ong_order_mgr.positions:
                print("  -> No ONG positions found")
        except Exception as _e:
            print(f"  ⚠️ ONG recovery failed: {_e}")

    try:
        while True:
            if not is_market_open():
                time.sleep(30)
                continue

            now = datetime.now()
            now_str = now.strftime("%H:%M:%S")
            today = now.date()
            now_t = now.hour * 100 + now.minute

            if current_day != today:
                if current_day is not None:
                    print(f"\n  [{now_str}] === DAY CHANGE {current_day} -> {today} ===")
                current_day = today
                order_mgr.daily_pnl = 0.0
                order_mgr.daily_trade_count = 0
                order_mgr.cooldown_until.clear()
                blacklist.daily_reset()
                ticker_exchange.clear()
                # Reset ONG/SM daily state flags
                ong_exit_done_today = False
                ong_entry_done_today = False
                sm_signal_done_today = False
                sm_entry_done_today = False
                sm_exit_done_today = False
                sm_long_tickers_pending = []
                sm_short_tickers_pending = []
                if sm_bar_builder is not None:
                    sm_bar_builder.reset()
                if ong_order_mgr is not None:
                    ong_order_mgr.daily_pnl = 0.0
                    ong_order_mgr.daily_trade_count = 0
                    ong_order_mgr.cooldown_until.clear()
                if sm_order_mgr is not None:
                    sm_order_mgr.daily_pnl = 0.0
                    sm_order_mgr.daily_trade_count = 0
                    sm_order_mgr.cooldown_until.clear()
                print(f"  [{now_str}] Daily PnL/cooldown/blacklist/exchange reset")

            # Force close
            if is_market_close_force():
                for pos in list(order_mgr.positions):
                    bars_df = bar_builder.get_bars(pos.ticker)
                    if not bars_df.empty:
                        current = float(bars_df["close"].iloc[-1])
                    else:
                        current = pos.entry_price
                    print(f"  [{now_str}] 引け強制決済: {pos.ticker} {pos.side}")
                    order_mgr.exit(pos, current, "引け強制決済")
                # SM failsafe: force-close any remaining SM positions at 15:20
                if sm_order_mgr:
                    for pos in list(sm_order_mgr.positions):
                        try:
                            board = client.get_board(pos.ticker)
                            current = float(board.get("CurrentPrice", pos.entry_price) or pos.entry_price) if board else pos.entry_price
                        except Exception:
                            current = pos.entry_price
                        print(f"  [{now_str}] [SM] 引け強制決済: {pos.ticker} {pos.side}")
                        sm_order_mgr.exit(pos, current, "SM引け強制決済")
            elif is_force_close_pm(pm_force_close):
                for pos in list(order_mgr.positions):
                    if hasattr(pos, 'entry_time') and pos.entry_time.hour >= 12:
                        bars_df = bar_builder.get_bars(pos.ticker)
                        if not bars_df.empty:
                            current = float(bars_df["close"].iloc[-1])
                        else:
                            current = pos.entry_price
                        print(f"  [{now_str}] PM時間決済({pm_force_close}): {pos.ticker} {pos.side}")
                        order_mgr.exit(pos, current, f"時間決済({pm_force_close})")

            # Fetch /board and build 5-min bars
            # Store Exchange from board (for info/display only — NOT for sendorder)
            for ticker in watchlist:
                try:
                    board = client.get_board(ticker)
                    if board and board.get("CurrentPrice"):
                        bar_builder.update(ticker, board)
                        if "Exchange" in board:
                            ticker_exchange[ticker] = int(board["Exchange"])
                except Exception as e:
                    pass
                time.sleep(0.3)

            # [SM] Poll /board for SM tickers during morning session (9:00-11:35)
            # Stop after SM entry is done (no SL/TP management for SM — exits strictly at 12:30)
            if sm_bar_builder is not None and not sm_entry_done_today and 900 <= now_t <= 1135:
                for ticker in sm_tickers_kabu:
                    try:
                        board = client.get_board(ticker)
                        if board and board.get("CurrentPrice"):
                            sm_bar_builder.update(ticker, board)
                    except Exception:
                        pass
                    time.sleep(0.1)

            # Position management
            for pos in list(order_mgr.positions):
                bars_df = bar_builder.get_bars(pos.ticker)
                if bars_df.empty or len(bars_df) < 2:
                    continue

                current = float(bars_df["close"].iloc[-1])
                trail_mult = strat_cfg["exit"]["trailing_atr_multiplier"]
                atr_period = strat_cfg["exit"].get("atr_period", 14)

                atr_val = calc_atr_from_bars(bars_df, atr_period)
                if atr_val is None:
                    atr_val = abs(float(bars_df["close"].iloc[-1]) - float(bars_df["close"].iloc[-2]))
                    if atr_val == 0:
                        atr_val = 1.0

                update_trailing_stop(pos, bars_df, trail_mult, atr_val)

                exit_reason = check_exit(pos, current)
                if exit_reason:
                    order_mgr.exit(pos, current, exit_reason)

            # Signal check
            if time.time() - last_signal_time >= live_cfg["interval"]["signal_check_sec"]:
                last_signal_time = time.time()
                now = datetime.now()
                now_str = now.strftime("%H:%M:%S")
                session = get_session_label(pm_entry_start, pm_entry_end)
                signal_check_count += 1

                morning = is_morning_entry()
                afternoon = is_afternoon_entry(pm_entry_start, pm_entry_end) and pm_available

                if not morning and not afternoon:
                    if signal_check_count % 10 == 1:
                        print(f"  [{now_str}] No entry window. Positions: {len(order_mgr.positions)}")
                else:
                    if signal_check_count % 5 == 1:
                        print(f"\n  [{now_str}] === [{session}] Signal Check #{signal_check_count} ===")
                        print(f"    Positions: {len(order_mgr.positions)}/{live_max_positions} "
                              f"| Exposure: {sum(p.entry_price*p.size for p in order_mgr.positions):,.0f}"
                              f"/{live_max_total_exposure:,.0f}")
                        for ticker in watchlist[:15]:
                            n_bars = bar_builder.get_bar_count(ticker)
                            bars_df = bar_builder.get_bars(ticker)
                            last_price = float(bars_df["close"].iloc[-1]) if not bars_df.empty else 0
                            ready = "OK" if n_bars >= MIN_BARS else "waiting"
                            bl_tag = " [BLOCKED]" if blacklist.is_blocked(ticker) else ""
                            exch = ticker_exchange.get(ticker, "?")
                            print(f"    {ticker}: price={last_price:.0f} bars={n_bars}/{MIN_BARS} "
                                  f"board_exchange={exch} order_exchange={DEFAULT_ORDER_EXCHANGE} [{ready}]{bl_tag}")
                        if len(watchlist) > 15:
                            print(f"    ... and {len(watchlist)-15} more stocks")

                    positions_full = len(order_mgr.positions) >= live_max_positions

                    if positions_full:
                        if signal_check_count % 5 == 1:
                            print(f"  [{now_str}] Positions full ({len(order_mgr.positions)}"
                                  f"/{live_max_positions}) - skipping all entries this cycle")

                    # AM candidates
                    am_candidates = []
                    if morning and not positions_full:
                        for ticker in watchlist:
                            if not order_mgr.can_entry(ticker):
                                continue
                            if blacklist.is_blocked(ticker):
                                continue

                            bars_df = bar_builder.get_bars(ticker)
                            n_bars = bar_builder.get_bar_count(ticker)
                            if n_bars < MIN_BARS:
                                continue

                            yf_ticker = ticker_map.get(ticker, f"{ticker}.T")
                            bias = daily_bias.get(yf_ticker, "NEUTRAL")
                            if bias == "BEAR":
                                continue

                            signal, score, detail = am_engine.evaluate_live(bars_df)

                            if score != 0:
                                print(f"  [{now_str}] [AM] {ticker}: score={score:.2f} signal={signal} bias={bias} ({detail})")

                            if signal in ("BUY", "SELL"):
                                current = float(bars_df["close"].iloc[-1])

                                vwap_val = None
                                if "vwap" in bars_df.columns:
                                    vwap_val = float(bars_df["vwap"].iloc[-1]) if not pd.isna(bars_df["vwap"].iloc[-1]) else None

                                if vwap_val is not None:
                                    if signal == "BUY" and current < vwap_val:
                                        print(f"  [{now_str}] [AM] {ticker}: BUY blocked by VWAP (price={current:.0f} < vwap={vwap_val:.0f})")
                                        continue
                                    if signal == "SELL" and current > vwap_val:
                                        print(f"  [{now_str}] [AM] {ticker}: SELL blocked by VWAP (price={current:.0f} > vwap={vwap_val:.0f})")
                                        continue

                                rsi_proxy = 50.0
                                if "rsi" in bars_df.columns and not pd.isna(bars_df["rsi"].iloc[-1]):
                                    rsi_proxy = float(bars_df["rsi"].iloc[-1])
                                priority = calc_priority_score(signal, {"rsi": rsi_proxy, "morning_move": 0.0})

                                am_candidates.append((ticker, signal, score, detail, bars_df, bias, priority))

                    am_candidates.sort(key=lambda x: x[6], reverse=True)

                    am_orders_placed = 0
                    am_break = False
                    for (ticker, signal, score, detail, bars_df, bias, priority) in am_candidates:
                        if am_orders_placed >= MAX_ORDERS_PER_CHECK_AM:
                            break
                        if am_break:
                            break
                        if len(order_mgr.positions) >= live_max_positions:
                            print(f"  [{now_str}] [AM] Positions full after {am_orders_placed} orders - stop")
                            break

                        current = float(bars_df["close"].iloc[-1])
                        atr_val = calc_atr_from_bars(bars_df, strat_cfg["exit"].get("atr_period", 14))
                        if atr_val is None:
                            atr_val = max(current * 0.01, 1)

                        sl_dist = atr_val * strat_cfg["exit"]["stop_loss_atr_multiplier"]
                        sl_dist = max(sl_dist, 1)
                        tp_dist = sl_dist * strat_cfg["exit"]["take_profit_rr_ratio"]

                        risk_per = strat_cfg["global"]["risk_per_trade"]
                        size = int((am_initial_capital * risk_per) / sl_dist)
                        size = max((size // 100) * 100, 100)

                        size = apply_live_risk_caps(
                            order_mgr.positions, current, size,
                            live_max_notional_per_position,
                            live_max_total_exposure,
                            "AM", ticker, now_str)
                        if size is None:
                            continue

                        entry_price = current
                        stop_loss, take_profit, trailing_stop = calc_entry_params(
                            signal, entry_price, sl_dist, tp_dist)

                        # v18.0: Always use DEFAULT_ORDER_EXCHANGE for new-open orders
                        # (board Exchange is for info only, not for sendorder)
                        exch = DEFAULT_ORDER_EXCHANGE
                        notional = entry_price * size
                        print(f"  [{now_str}] [AM] >>> ENTRY {signal} {ticker}: price={entry_price:.0f} "
                              f"size={size} notional={notional:,.0f} exchange={exch} "
                              f"(per_pos_cap={live_max_notional_per_position:,.0f} "
                              f"total_cap={live_max_total_exposure:,.0f}) "
                              f"SL={stop_loss:.0f} TP={take_profit:.0f} priority={priority:.2f}")

                        reason = f"AM Ensemble score={score:.2f} bias={bias} ({detail})"
                        action = execute_entry_with_error_handling(
                            order_mgr, ticker, signal, entry_price, size,
                            stop_loss, take_profit, reason, "AM",
                            blacklist, now_str,
                            exchange=exch,
                        )

                        if action == "break":
                            am_break = True
                            break
                        elif action in ("ok", "retry_ok"):
                            am_orders_placed += 1

                    # PM candidates
                    pm_candidates = []
                    positions_full = len(order_mgr.positions) >= live_max_positions
                    if afternoon and not am_break and not positions_full:
                        for ticker in watchlist:
                            if not order_mgr.can_entry(ticker):
                                continue
                            if blacklist.is_blocked(ticker):
                                if signal_check_count % 5 == 1:
                                    print(f"  [{now_str}] [PM] {ticker}: SKIP (blocked: {blacklist.get_reason(ticker)})")
                                continue

                            bars_df = bar_builder.get_bars(ticker)
                            n_bars = bar_builder.get_bar_count(ticker)
                            if n_bars < MIN_BARS:
                                continue

                            signal, info = pm_engine.evaluate_live(bars_df)
                            rsi_val = info.get("rsi")
                            mm_val = info.get("morning_move")

                            if rsi_val is not None and (rsi_val < 35 or rsi_val > 65):
                                print(f"  [{now_str}] [PM] {ticker}: RSI={rsi_val:.1f} morning_move={mm_val:.1f}% signal={signal}")

                            if signal in ("BUY", "SELL"):
                                priority = calc_priority_score(signal, info)
                                pm_candidates.append((ticker, signal, info, bars_df, priority))

                    pm_candidates.sort(key=lambda x: x[4], reverse=True)

                    pm_orders_placed = 0
                    for (ticker, signal, info, bars_df, priority) in pm_candidates:
                        if pm_orders_placed >= MAX_ORDERS_PER_CHECK_PM:
                            break
                        if len(order_mgr.positions) >= live_max_positions:
                            print(f"  [{now_str}] [PM] Positions full after {pm_orders_placed} orders - stop")
                            break

                        current = float(bars_df["close"].iloc[-1])
                        rsi_val = info.get("rsi")
                        mm_val = info.get("morning_move")

                        pm_exit = afternoon_cfg["exit"]
                        atr_val = calc_atr_from_bars(bars_df, pm_exit.get("atr_period", 14))
                        if atr_val is None:
                            atr_val = max(current * 0.01, 1)

                        sl_dist = atr_val * pm_exit["stop_loss_atr_multiplier"]
                        sl_dist = max(sl_dist, 1)
                        tp_dist = atr_val * pm_exit["take_profit_atr_multiplier"]

                        pm_global = afternoon_cfg["global"]
                        risk_per = pm_global["risk_per_trade"]
                        size = int((pm_initial_capital * risk_per) / sl_dist)
                        size = max((size // 100) * 100, 100)

                        size = apply_live_risk_caps(
                            order_mgr.positions, current, size,
                            live_max_notional_per_position,
                            live_max_total_exposure,
                            "PM", ticker, now_str)
                        if size is None:
                            continue

                        entry_price = current
                        stop_loss, take_profit, trailing_stop = calc_entry_params(
                            signal, entry_price, sl_dist, tp_dist)

                        # v18.0: Always use DEFAULT_ORDER_EXCHANGE for new-open orders
                        exch = DEFAULT_ORDER_EXCHANGE
                        notional = entry_price * size
                        print(f"  [{now_str}] [PM] >>> ENTRY {signal} {ticker}: price={entry_price:.0f} "
                              f"size={size} notional={notional:,.0f} exchange={exch} "
                              f"(per_pos_cap={live_max_notional_per_position:,.0f} "
                              f"total_cap={live_max_total_exposure:,.0f}) "
                              f"SL={stop_loss:.0f} TP={take_profit:.0f} "
                              f"RSI={rsi_val:.1f} priority={priority:.2f}")

                        reason = f"PM Reversal RSI={rsi_val:.1f} mm={mm_val:.1f}%"
                        action = execute_entry_with_error_handling(
                            order_mgr, ticker, signal, entry_price, size,
                            stop_loss, take_profit, reason, "PM",
                            blacklist, now_str,
                            exchange=exch,
                        )

                        if action == "break":
                            break
                        elif action in ("ok", "retry_ok"):
                            pm_orders_placed += 1

            # ──────────────────────────────────────────────────────────────
            # ONG/SM Phase Checks (time-sensitive, one-time per day)
            # ──────────────────────────────────────────────────────────────

            # [ONG EXIT] 9:00-9:05: 前日ONG翌朝成行決済
            if ong_order_mgr and not ong_exit_done_today and is_ong_exit_window():
                ong_exit_done_today = True
                if ong_order_mgr.positions:
                    print(f"\n  [{now_str}] === [ONG EXIT] 翌朝成行決済 "
                          f"({len(ong_order_mgr.positions)} positions) ===")
                    for pos in list(ong_order_mgr.positions):
                        try:
                            _board = client.get_board(pos.ticker)
                            _cur = (float(_board.get("CurrentPrice", pos.entry_price)
                                         or pos.entry_price)
                                    if _board else pos.entry_price)
                        except Exception:
                            _cur = pos.entry_price
                        print(f"  [{now_str}] [ONG EXIT] {pos.ticker}: "
                              f"current={_cur:.0f} entry={pos.entry_price:.0f}")
                        ong_order_mgr.exit(pos, _cur, "ONG翌朝成行決済")
                else:
                    print(f"  [{now_str}] [ONG EXIT] No ONG positions to close")

            # [SM SIGNAL] 11:25: 前場シグナル評価（1回のみ）
            if sm_engine and sm_bar_builder and not sm_signal_done_today and now_t >= 1125:
                sm_signal_done_today = True
                print(f"\n  [{now_str}] === [SM SIGNAL] 前場シグナル評価 ===")
                _morning_data: dict[str, pd.DataFrame] = {}
                for _ticker in sm_tickers_kabu:
                    _bars = sm_bar_builder.get_bars(_ticker)
                    if _bars.empty:
                        continue
                    # Filter to current morning (bars up to 11:25)
                    _mask = (_bars.index.hour * 100 + _bars.index.minute) <= 1125
                    _mb = _bars[_mask]
                    if not _mb.empty:
                        _morning_data[_ticker] = _mb
                if _morning_data:
                    _sm_result = sm_engine.generate_daily_signal(_morning_data, {})
                    if _sm_result:
                        sm_long_tickers_pending, sm_short_tickers_pending = _sm_result
                        print(f"  [SM] LONG:  {sm_long_tickers_pending}")
                        print(f"  [SM] SHORT: {sm_short_tickers_pending}")
                    else:
                        sm_long_tickers_pending = []
                        sm_short_tickers_pending = []
                        print("  [SM] No signals generated")
                else:
                    sm_long_tickers_pending = []
                    sm_short_tickers_pending = []
                    print("  [SM] No morning bar data available")

            # [SM ENTRY] 11:30: SM成行発注（1回のみ）
            if sm_order_mgr and sm_signal_done_today and not sm_entry_done_today and now_t >= 1130:
                sm_entry_done_today = True
                _sm_capital = float(sm_cfg["global"]["initial_capital"])
                _sm_risk = float(sm_cfg["global"]["risk_per_position"])
                print(f"\n  [{now_str}] === [SM ENTRY] 発注 "
                      f"(long={len(sm_long_tickers_pending)} "
                      f"short={len(sm_short_tickers_pending)}) ===")
                for _side, _tickers in (("BUY", sm_long_tickers_pending),
                                        ("SELL", sm_short_tickers_pending)):
                    for _ticker in _tickers:
                        if not sm_order_mgr.can_entry(_ticker):
                            print(f"  [SM] SKIP {_ticker}: can_entry=False")
                            continue
                        try:
                            _board = client.get_board(_ticker)
                            _price = float(_board.get("CurrentPrice", 0) or 0) if _board else 0
                            if _price <= 0:
                                print(f"  [SM] SKIP {_ticker}: invalid price={_price}")
                                continue
                            # SM: size = capital × risk_per_position / entry_price
                            _size = int(_sm_capital * _sm_risk / _price)
                            _size = max((_size // 100) * 100, 100)
                            # Placeholder SL/TP (wide ±15%) — SM never hits them.
                            # All SM positions are closed unconditionally at 12:30
                            # via [SM EXIT] block, not via SL/TP logic.
                            _sl = _price * 0.85 if _side == "BUY" else _price * 1.15
                            _tp = _price * 1.15 if _side == "BUY" else _price * 0.85
                            _notional = _price * _size
                            print(f"  [{now_str}] [SM] >>> ENTRY {_side} {_ticker}: "
                                  f"price={_price:.0f} size={_size} notional={_notional:,.0f}")
                            sm_order_mgr.entry(
                                _ticker, _side, _price, _size,
                                stop_loss=_sl, take_profit=_tp,
                                reason=f"SM順張り{'ロング' if _side == 'BUY' else 'ショート'}",
                                session="SM",
                                exchange=DEFAULT_ORDER_EXCHANGE,
                            )
                            time.sleep(ORDER_INTERVAL_SEC)
                        except Exception as _e:
                            print(f"  ⚠️ [SM] ENTRY {_side} {_ticker} error: {_e}")

            # [SM EXIT] 12:30: SM後場寄り全決済（1回のみ）
            if sm_order_mgr and not sm_exit_done_today and now_t >= 1230:
                sm_exit_done_today = True
                if sm_order_mgr.positions:
                    print(f"\n  [{now_str}] === [SM EXIT] 後場寄り全決済 "
                          f"({len(sm_order_mgr.positions)} positions) ===")
                    for pos in list(sm_order_mgr.positions):
                        try:
                            _board = client.get_board(pos.ticker)
                            _cur = (float(_board.get("CurrentPrice", pos.entry_price)
                                         or pos.entry_price)
                                    if _board else pos.entry_price)
                        except Exception:
                            _cur = pos.entry_price
                        print(f"  [{now_str}] [SM EXIT] {pos.ticker} {pos.side}: "
                              f"current={_cur:.0f}")
                        sm_order_mgr.exit(pos, _cur, "SM後場寄り成行決済")

            # [ONG ENTRY] 14:50-15:00: ONG引け前シグナル判定→成行買い（1回のみ）
            if ong_order_mgr and not ong_entry_done_today and is_ong_entry_window():
                ong_entry_done_today = True
                _today_date = now.date()
                _skip_friday = ong_cfg["ong"].get("skip_friday", True)
                _ong_max_pos = ong_order_mgr.trade_config["max_positions"]
                _ong_max_risk = float(ong_cfg["global"]["max_risk_per_trade"])
                _atr_period = int(ong_cfg["ong"].get("atr_period", 14))
                _atr_mult = float(ong_cfg["ong"].get("atr_risk_multiplier", 2.0))
                _sl_pct = float(ong_cfg["ong"].get("stop_loss_pct", -1.0)) / 100.0

                if _skip_friday and _today_date.weekday() == 4:
                    print(f"\n  [{now_str}] [ONG] 本日は金曜日: エントリースキップ")
                elif _today_date in BLACKOUT_DATES:
                    print(f"\n  [{now_str}] [ONG] 本日はブラックアウト日: エントリースキップ")
                else:
                    print(f"\n  [{now_str}] === [ONG ENTRY] 引け前シグナル判定 ===")
                    for _ticker in ong_tickers_kabu:
                        if len(ong_order_mgr.positions) >= _ong_max_pos:
                            print(f"  [ONG] Max positions reached ({_ong_max_pos})")
                            break
                        if _ticker not in ong_daily_data:
                            continue
                        if blacklist.is_blocked(_ticker):
                            continue
                        try:
                            _board = client.get_board(_ticker)
                            if not _board:
                                continue
                            _o = float(_board.get("OpeningPrice", 0) or 0)
                            _h = float(_board.get("HighPrice", 0) or 0)
                            _l = float(_board.get("LowPrice", 0) or 0)
                            _c = float(_board.get("CurrentPrice", 0) or 0)
                            _v = float(_board.get("TradingVolume", 0) or 0)
                            if any(x <= 0 for x in [_o, _h, _l, _c]):
                                continue
                            # Append today's row to historical data
                            _today_ts = pd.Timestamp(_today_date)
                            _today_row = pd.DataFrame(
                                {"open": [_o], "high": [_h], "low": [_l],
                                 "close": [_c], "volume": [_v]},
                                index=[_today_ts],
                            )
                            _full_df = pd.concat([ong_daily_data[_ticker], _today_row])
                            # Pass nikkei_etf_df=None to skip the night-tailwind condition:
                            # next-day ETF open is not yet known at 14:50, so we rely on
                            # IBS + RSI(2) + decline-rate (3-condition live mode per spec)
                            _sig_result = generate_ong_signals(
                                {_ticker: _full_df}, None, ong_cfg
                            )
                            if _ticker not in _sig_result:
                                continue
                            _df_sig = _sig_result[_ticker]
                            _last = _df_sig.iloc[-1]
                            _ibs = float(_last.get("ibs", float("nan")))
                            _rsi2 = float(_last.get("rsi2", float("nan")))
                            _dpct = float(_last.get("day_return_pct", float("nan")))
                            if not _last.get("ONG_signal", False):
                                print(f"  [ONG] {_ticker}: No signal "
                                      f"(IBS={_ibs:.3f} RSI2={_rsi2:.1f} ret={_dpct:.1f}%)")
                                continue
                            # Calculate ATR for position sizing
                            _atr_s = ta.atr(_df_sig["high"], _df_sig["low"],
                                            _df_sig["close"], length=_atr_period)
                            if _atr_s is None or pd.isna(_atr_s.iloc[-1]):
                                print(f"  [ONG] {_ticker}: Signal but ATR unavailable — skip")
                                continue
                            _atr_val = float(_atr_s.iloc[-1])
                            if _atr_val <= 0:
                                continue
                            # ONG: size = max_risk_per_trade / (ATR × atr_risk_multiplier)
                            _size = int(_ong_max_risk / (_atr_val * _atr_mult))
                            _size = max((_size // 100) * 100, 100)
                            # SL = close × (1 + stop_loss_pct/100), e.g. -1% → close × 0.99
                            # ONG never hits SL via position-management loop — exits at open next day
                            _sl = _c * (1.0 + _sl_pct)
                            _tp = _c * 1.05             # placeholder TP (exit at next day open)
                            _notional = _c * _size
                            print(f"  [{now_str}] [ONG] ✅ Signal: {_ticker} "
                                  f"IBS={_ibs:.3f} RSI2={_rsi2:.1f} ret={_dpct:.1f}% "
                                  f"ATR={_atr_val:.0f} size={_size} "
                                  f"notional={_notional:,.0f} close={_c:.0f}")
                            print(f"  [{now_str}] [ONG] >>> ENTRY BUY {_ticker}: "
                                  f"price={_c:.0f} size={_size} "
                                  f"MTT={ONG_MARGIN_TRADE_TYPE} (一般信用長期)")
                            ong_order_mgr.entry(
                                _ticker, "BUY", _c, _size,
                                stop_loss=_sl, take_profit=_tp,
                                reason=(f"ONG IBS={_ibs:.3f} RSI2={_rsi2:.1f} "
                                        f"ret={_dpct:.1f}%"),
                                session="ONG",
                                exchange=DEFAULT_ORDER_EXCHANGE,
                            )
                            time.sleep(ORDER_INTERVAL_SEC)
                        except Exception as _e:
                            print(f"  ⚠️ [ONG] {_ticker} error: {_e}")

            time.sleep(live_cfg["interval"]["price_check_sec"])

    except KeyboardInterrupt:
        print("\nStopped.")
        print(f"  Total signal checks: {signal_check_count}")
        print(f"  Open positions: {len(order_mgr.positions)}")
        if order_mgr.trades:
            print(order_mgr.get_daily_summary())
        if ong_order_mgr and (ong_order_mgr.trades or ong_order_mgr.positions):
            ong_pnl = sum(t.pnl for t in ong_order_mgr.trades)
            print(f"  [ONG] trades={len(ong_order_mgr.trades)} PnL={ong_pnl:+,.0f}円 "
                  f"open={len(ong_order_mgr.positions)}")
        if sm_order_mgr and sm_order_mgr.trades:
            sm_pnl = sum(t.pnl for t in sm_order_mgr.trades)
            print(f"  [SM]  trades={len(sm_order_mgr.trades)} PnL={sm_pnl:+,.0f}円")


if __name__ == "__main__":
    main()
