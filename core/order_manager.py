"""Order / position manager (v18.0: multi-lot hold_entries, spec-aligned close)"""

import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from core.api_client import KabuClient

# MarginTradeType labels for logging
MARGIN_TRADE_TYPE_LABELS = {
    1: "制度信用",
    2: "一般(長期)",
    3: "一般(デイトレ)",
}

# Delay (seconds) before polling /positions after new order accepted
_FILL_POLL_DELAY = 2.0
# Max attempts to poll for fill
_FILL_POLL_MAX = 5
# Interval between poll attempts
_FILL_POLL_INTERVAL = 1.5


@dataclass
class HoldEntry:
    """Single lot from /positions (ExecutionID-based).

    Represents one execution fill. A single order may produce multiple
    HoldEntry records when the order is filled in multiple lots (split fills).
    """
    hold_id: str       # ExecutionID from /positions (used as HoldID in ClosePositions)
    qty: int           # LeavesQty from /positions
    exchange: int      # Exchange from /positions
    price: float = 0.0 # Price from /positions (informational)


@dataclass
class LivePosition:
    """Open position"""
    ticker: str
    side: str              # "BUY" or "SELL"
    entry_price: float
    entry_time: datetime
    size: int
    stop_loss: float
    take_profit: float
    trailing_stop: float
    exchange: int = 1      # board-derived exchange code (for INFO queries)
    order_exchange: int = 27  # exchange used in /sendorder (for close order)
    hold_entries: list = field(default_factory=list)  # list[HoldEntry]
    order_id: str = ""     # order ID from sendorder response
    reason: str = ""
    session: str = ""      # "AM" or "PM"

    @property
    def hold_id(self) -> str:
        """Backward-compatible: return first hold_id or empty string."""
        if self.hold_entries:
            return self.hold_entries[0].hold_id
        return ""

    @property
    def position_exchange(self) -> int:
        """Backward-compatible: return first exchange or 0."""
        if self.hold_entries:
            return self.hold_entries[0].exchange
        return 0

    @property
    def total_hold_qty(self) -> int:
        """Total qty across all hold entries."""
        return sum(e.qty for e in self.hold_entries)


@dataclass
class LiveTrade:
    """Closed trade"""
    ticker: str
    side: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    size: int
    pnl: float
    reason: str = ""
    session: str = ""


class OrderManager:
    """Order / position manager (v18.0: multi-lot hold_entries + spec-aligned close)"""

    def __init__(self, client: KabuClient, config: dict):
        self.client = client
        self.paper_mode = config["mode"]["paper_trade"]
        self.trade_config = config["trade"]
        self.positions: list[LivePosition] = []
        self.trades: list[LiveTrade] = []
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.cooldown_until: dict[str, datetime] = {}

        # margin_trade_type from config (default: 3 = day-trade)
        self.margin_trade_type: int = int(self.trade_config.get("margin_trade_type", 3))

        # BT-aligned cooldown config (set by main_live after init)
        self.cooldown_config: dict = {
            "am_enabled": False,
            "am_loss_min": 75,
            "am_win_min": 25,
            "pm_enabled": False,
            "pm_loss_min": 30,
            "pm_win_min": 10,
        }

    def can_entry(self, ticker: str) -> bool:
        """Check if entry is allowed"""
        if len(self.positions) >= self.trade_config["max_positions"]:
            return False
        if any(p.ticker == ticker for p in self.positions):
            return False
        max_loss = self.trade_config["initial_capital"] * self.trade_config["max_daily_loss"]
        if self.daily_pnl < 0 and abs(self.daily_pnl) >= max_loss:
            return False
        if ticker in self.cooldown_until:
            if datetime.now() < self.cooldown_until[ticker]:
                return False
        return True

    def _poll_hold_entries(self, ticker: str, side: str) -> list:
        """Poll /positions to resolve hold entries (ExecutionIDs) after order fill.

        Returns:
            list[HoldEntry] — all matching lots, or [] if not found.
        """
        time.sleep(_FILL_POLL_DELAY)

        for attempt in range(1, _FILL_POLL_MAX + 1):
            hold_infos = self.client.resolve_position_hold_ids(ticker, side)
            if hold_infos:
                entries = [
                    HoldEntry(
                        hold_id=info["hold_id"],
                        qty=info["qty"],
                        exchange=info["exchange"],
                        price=info.get("price", 0.0),
                    )
                    for info in hold_infos
                ]
                total_qty = sum(e.qty for e in entries)
                ids_str = ", ".join(
                    f"{e.hold_id}(x{e.qty}@Ex{e.exchange})" for e in entries
                )
                print(f"  ✅ hold_entries resolved: {ticker} -> {len(entries)} lot(s), "
                      f"total_qty={total_qty} [{ids_str}] (attempt {attempt})")
                return entries

            if attempt < _FILL_POLL_MAX:
                print(f"  ⏳ hold_entries not yet available for {ticker} "
                      f"(attempt {attempt}/{_FILL_POLL_MAX})")
                time.sleep(_FILL_POLL_INTERVAL)

        print(f"  ⚠️ hold_entries NOT resolved for {ticker} after {_FILL_POLL_MAX} attempts")
        return []

    def entry(self, ticker: str, side: str, price: float,
              size: int, stop_loss: float, take_profit: float,
              reason: str = "", session: str = "",
              exchange: int = 27) -> dict:
        """New entry.

        Args:
            exchange: Exchange for /sendorder new-open (27=東証+ by default).

        Returns:
            dict: Structured result.
        """

        now = datetime.now()

        if self.paper_mode:
            pos = LivePosition(
                ticker=ticker, side=side,
                entry_price=price, entry_time=now,
                size=size, stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=stop_loss,
                exchange=exchange,
                order_exchange=exchange,
                hold_entries=[HoldEntry(
                    hold_id=f"PAPER_{ticker}_{int(time.time())}",
                    qty=size, exchange=exchange,
                )],
                reason=reason,
                session=session,
            )
            self.positions.append(pos)
            self.daily_trade_count += 1
            mtt_label = MARGIN_TRADE_TYPE_LABELS.get(self.margin_trade_type, "?")
            print(f"  📝 [PAPER] {side} {ticker} × {size}株 @ {price:.0f}円 "
                  f"Exchange={exchange} MTT={self.margin_trade_type}({mtt_label})")
            print(f"       SL={stop_loss:.0f} TP={take_profit:.0f} | {reason}")
            return {"ok": True}
        else:
            mtt = self.margin_trade_type
            mtt_label = MARGIN_TRADE_TYPE_LABELS.get(mtt, "?")

            result = self.client.send_margin_order(
                symbol=ticker,
                exchange=exchange,
                side=side,
                qty=size,
                order_type=1,  # market
                margin_trade_type=mtt,
            )

            if result and result.get("ok"):
                order_id = result.get("order_id", "")

                # Poll /positions to get ExecutionIDs (hold_entries) for later close
                hold_entries = self._poll_hold_entries(ticker, side)

                pos = LivePosition(
                    ticker=ticker, side=side,
                    entry_price=price, entry_time=now,
                    size=size, stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop=stop_loss,
                    exchange=exchange,
                    order_exchange=exchange,
                    order_id=order_id,
                    hold_entries=hold_entries if hold_entries else [],
                    reason=reason,
                    session=session,
                )
                self.positions.append(pos)
                self.daily_trade_count += 1
                hold_id_display = pos.hold_id or "(none)"
                n_lots = len(hold_entries)
                print(f"  🔥 [LIVE] {side} {ticker} × {size}株 "
                      f"Exchange={exchange} MTT={mtt}({mtt_label}) "
                      f"| OrderID={order_id} HoldID={hold_id_display} lots={n_lots}")
                return result
            else:
                code = result.get('code', 0) if result else '?'
                print(f"  ❌ 発注失敗: {ticker} code={code} "
                      f"Exchange={exchange} MTT={mtt}({mtt_label})")
                if result:
                    return result
                return {"ok": False, "http": 0, "code": 0, "message": "no result from api"}

    def _try_resolve_hold_entries(self, pos: LivePosition) -> None:
        """Attempt to resolve hold_entries if they were not obtained during entry."""
        if pos.hold_entries and not pos.hold_entries[0].hold_id.startswith("PAPER"):
            return  # already resolved

        print(f"  🔄 Attempting late hold_entries resolution for {pos.ticker}...")
        entries = self._poll_hold_entries(pos.ticker, pos.side)
        if entries:
            pos.hold_entries = entries

    @staticmethod
    def _build_close_positions(hold_entries: list, close_qty: int) -> tuple:
        """Build ClosePositions list from hold_entries for the requested close_qty.

        Allocates lots in order until close_qty is fulfilled.

        Returns:
            (close_positions_list, consumed) or (None, None) if insufficient qty.
            close_positions_list: list of {"HoldID": str, "Qty": int}
            consumed: list of (index, allocated_qty) for updating hold_entries
        """
        total_available = sum(e.qty for e in hold_entries)
        if close_qty > total_available:
            return None, None

        close_positions = []
        consumed = []  # (index, allocated_qty)
        remaining = close_qty

        for i, entry in enumerate(hold_entries):
            if remaining <= 0:
                break
            alloc = min(entry.qty, remaining)
            close_positions.append({"HoldID": entry.hold_id, "Qty": alloc})
            consumed.append((i, alloc))
            remaining -= alloc

        if remaining > 0:
            return None, None

        return close_positions, consumed

    @staticmethod
    def _resolve_close_exchange(hold_entries: list) -> tuple[int | None, str]:
        """Determine Exchange for close order from hold_entries.

        Returns:
            (exchange, error_message)
            - If all entries share the same Exchange: (exchange_value, "")
            - If Exchanges are mixed: (None, error_description)
            - If no entries: (None, error_description)
        """
        if not hold_entries:
            return None, "no hold_entries"

        exchanges = set(e.exchange for e in hold_entries)
        if len(exchanges) == 1:
            return exchanges.pop(), ""
        else:
            ex_str = ", ".join(str(ex) for ex in sorted(exchanges))
            return None, f"mixed Exchanges in hold_entries: {{{ex_str}}}"

    def exit(self, pos: LivePosition, current_price: float, reason: str) -> LiveTrade:
        """Close position.

        In LIVE mode:
        - Builds ClosePositions from multiple hold_entries
        - Only removes local position if API close succeeds
        - Uses hold_entries (ExecutionIDs) from /positions
        - Exchange is derived from hold_entries (must be uniform across all lots)
        - If hold_entries have mixed Exchanges, close is refused (error)
        - DelivType=0 for new-open, DelivType=2 for close (asymmetric rule)
        """

        now = datetime.now()

        if pos.side == "BUY":
            pnl = (current_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - current_price) * pos.size

        trade = LiveTrade(
            ticker=pos.ticker, side=pos.side,
            entry_price=pos.entry_price,
            exit_price=current_price,
            entry_time=pos.entry_time,
            exit_time=now,
            size=pos.size, pnl=pnl,
            reason=reason,
            session=pos.session,
        )

        if not self.paper_mode:
            # Try to resolve hold_entries if missing
            self._try_resolve_hold_entries(pos)

            if not pos.hold_entries or pos.hold_entries[0].hold_id.startswith("PAPER"):
                print(f"  ❌ [LIVE] Cannot close {pos.ticker}: no hold_entries (ExecutionIDs)")
                print(f"     Position remains open. Manual intervention required.")
                # DO NOT remove from self.positions — position is still open on exchange
                return trade

            # Build ClosePositions from hold_entries
            close_positions, consumed = self._build_close_positions(
                pos.hold_entries, pos.size
            )

            if close_positions is None:
                total_hold = pos.total_hold_qty
                print(f"  ❌ [LIVE] Cannot close {pos.ticker}: insufficient hold qty "
                      f"(need={pos.size}, have={total_hold})")
                print(f"     Position remains open. Manual intervention required.")
                # DO NOT remove from self.positions — insufficient qty
                return trade

            close_side = "SELL" if pos.side == "BUY" else "BUY"

            # Derive Exchange from hold_entries (must be uniform)
            close_exchange, ex_error = self._resolve_close_exchange(pos.hold_entries)
            if close_exchange is None:
                print(f"  ❌ [LIVE] Cannot close {pos.ticker}: {ex_error}")
                print(f"     Position remains open. Manual intervention required.")
                # DO NOT remove from self.positions — mixed Exchange
                return trade

            close_result = self.client.send_margin_close(
                symbol=pos.ticker,
                exchange=close_exchange,
                side=close_side,
                qty=pos.size,
                close_positions=close_positions,
                order_type=1,
                margin_trade_type=self.margin_trade_type,
            )

            if not close_result or not close_result.get("ok"):
                code = close_result.get("code", "?") if close_result else "?"
                msg = close_result.get("message", "") if close_result else ""
                cp_str = ", ".join(
                    f"{cp['HoldID']}(x{cp['Qty']})" for cp in close_positions
                )
                print(f"  ❌ [LIVE] Close order FAILED for {pos.ticker}: "
                      f"code={code} {msg}")
                print(f"     Exchange={close_exchange} ClosePositions=[{cp_str}]")
                print(f"     Position remains open. Manual intervention required.")
                # DO NOT remove from self.positions — API close failed
                return trade

            # API success → update local hold_entries (consume allocated qty)
            for idx, alloc_qty in consumed:
                pos.hold_entries[idx].qty -= alloc_qty
            # Remove fully consumed entries
            pos.hold_entries = [e for e in pos.hold_entries if e.qty > 0]

            cp_str = ", ".join(
                f"{cp['HoldID']}(x{cp['Qty']})" for cp in close_positions
            )
            print(f"  ✅ [LIVE] Close order accepted for {pos.ticker}: "
                  f"OrderID={close_result.get('order_id', '')} "
                  f"Exchange={close_exchange} ClosePositions=[{cp_str}]")

        mode_tag = "PAPER" if self.paper_mode else "LIVE"
        pnl_str = f"+{pnl:,.0f}" if pnl >= 0 else f"{pnl:,.0f}"
        print(f"  {'✅' if pnl >= 0 else '❌'} [{mode_tag}] 決済 {pos.ticker} [{pos.session}] | {pnl_str}円 | {reason}")

        self.positions.remove(pos)
        self.trades.append(trade)
        self.daily_pnl += pnl

        # BT-aligned cooldown
        session = pos.session or "AM"
        cd_cfg = self.cooldown_config

        if session == "PM" and cd_cfg.get("pm_enabled", False):
            if pnl < 0:
                cd_minutes = cd_cfg.get("pm_loss_min", 30)
            elif cd_cfg.get("pm_win_min", 0) > 0:
                cd_minutes = cd_cfg.get("pm_win_min", 10)
            else:
                cd_minutes = 0
            if cd_minutes > 0:
                self.cooldown_until[pos.ticker] = now + timedelta(minutes=cd_minutes)
        elif cd_cfg.get("am_enabled", False):
            if pnl < 0:
                cd_minutes = cd_cfg.get("am_loss_min", 75)
            elif cd_cfg.get("am_win_min", 0) > 0:
                cd_minutes = cd_cfg.get("am_win_min", 25)
            else:
                cd_minutes = 0
            if cd_minutes > 0:
                self.cooldown_until[pos.ticker] = now + timedelta(minutes=cd_minutes)

        return trade

    def get_daily_summary(self) -> str:
        """Daily summary"""
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self.trades)

        am_trades = [t for t in self.trades if t.session == "AM"]
        pm_trades = [t for t in self.trades if t.session == "PM"]
        am_pnl = sum(t.pnl for t in am_trades)
        pm_pnl = sum(t.pnl for t in pm_trades)

        summary = f"""
日次サマリー
  トレード数: {len(self.trades)}
  勝ち: {len(wins)} / 負け: {len(losses)}
  損益: {total_pnl:+,.0f}円
    AM: {len(am_trades)}件 -> {am_pnl:+,.0f}円
    PM: {len(pm_trades)}件 -> {pm_pnl:+,.0f}円
  残ポジション: {len(self.positions)}
"""
        return summary
