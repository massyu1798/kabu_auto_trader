"""Order / position manager (v17.0: hold_id from /positions, spec-aligned close)"""

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
    hold_id: str = ""      # ExecutionID from /positions (for ClosePositions)
    position_exchange: int = 0  # Exchange from /positions (for close order)
    order_id: str = ""     # order ID from sendorder response
    reason: str = ""
    session: str = ""      # "AM" or "PM"


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
    """Order / position manager (v17.0: hold_id + spec-aligned close)"""

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

    def _poll_hold_id(self, ticker: str, side: str) -> tuple[str, int]:
        """Poll /positions to resolve hold_id (ExecutionID) after order fill.

        Returns:
            (hold_id, position_exchange) or ("", 0) if not found.
        """
        time.sleep(_FILL_POLL_DELAY)

        for attempt in range(1, _FILL_POLL_MAX + 1):
            hold_infos = self.client.resolve_position_hold_ids(ticker, side)
            if hold_infos:
                # Use the latest (last) position entry
                info = hold_infos[-1]
                hold_id = info["hold_id"]
                pos_exchange = info["exchange"]
                print(f"  ✅ hold_id resolved: {ticker} -> ExecutionID={hold_id} "
                      f"Exchange={pos_exchange} (attempt {attempt})")
                return hold_id, pos_exchange

            if attempt < _FILL_POLL_MAX:
                print(f"  ⏳ hold_id not yet available for {ticker} "
                      f"(attempt {attempt}/{_FILL_POLL_MAX})")
                time.sleep(_FILL_POLL_INTERVAL)

        print(f"  ⚠️ hold_id NOT resolved for {ticker} after {_FILL_POLL_MAX} attempts")
        return "", 0

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
                hold_id=f"PAPER_{ticker}_{int(time.time())}",
                position_exchange=exchange,
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

                # Poll /positions to get ExecutionID (hold_id) for later close
                hold_id, pos_exchange = self._poll_hold_id(ticker, side)

                pos = LivePosition(
                    ticker=ticker, side=side,
                    entry_price=price, entry_time=now,
                    size=size, stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop=stop_loss,
                    exchange=exchange,
                    order_exchange=exchange,
                    order_id=order_id,
                    hold_id=hold_id,
                    position_exchange=pos_exchange if pos_exchange else exchange,
                    reason=reason,
                    session=session,
                )
                self.positions.append(pos)
                self.daily_trade_count += 1
                print(f"  🔥 [LIVE] {side} {ticker} × {size}株 "
                      f"Exchange={exchange} MTT={mtt}({mtt_label}) "
                      f"| OrderID={order_id} HoldID={hold_id}")
                return result
            else:
                code = result.get('code', 0) if result else '?'
                print(f"  ❌ 発注失敗: {ticker} code={code} "
                      f"Exchange={exchange} MTT={mtt}({mtt_label})")
                if result:
                    return result
                return {"ok": False, "http": 0, "code": 0, "message": "no result from api"}

    def _try_resolve_hold_id(self, pos: LivePosition) -> None:
        """Attempt to resolve hold_id if it was not obtained during entry."""
        if pos.hold_id and not pos.hold_id.startswith("PAPER"):
            return  # already resolved

        print(f"  🔄 Attempting late hold_id resolution for {pos.ticker}...")
        hold_id, pos_exchange = self._poll_hold_id(pos.ticker, pos.side)
        if hold_id:
            pos.hold_id = hold_id
            pos.position_exchange = pos_exchange

    def exit(self, pos: LivePosition, current_price: float, reason: str) -> LiveTrade:
        """Close position.

        In LIVE mode:
        - Sends margin close order via API
        - Only removes local position if API close succeeds
        - Uses hold_id (ExecutionID) and position_exchange from /positions
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
            # Try to resolve hold_id if missing
            self._try_resolve_hold_id(pos)

            if not pos.hold_id or pos.hold_id.startswith("PAPER"):
                print(f"  ❌ [LIVE] Cannot close {pos.ticker}: no hold_id (ExecutionID)")
                print(f"     Position remains open. Manual intervention required.")
                # DO NOT remove from self.positions — position is still open on exchange
                return trade

            close_side = "SELL" if pos.side == "BUY" else "BUY"

            # Use the Exchange from /positions (matches the actual position)
            close_exchange = pos.position_exchange if pos.position_exchange else pos.order_exchange

            close_result = self.client.send_margin_close(
                symbol=pos.ticker,
                exchange=close_exchange,
                side=close_side,
                qty=pos.size,
                hold_id=pos.hold_id,
                order_type=1,
                margin_trade_type=self.margin_trade_type,
            )

            if not close_result or not close_result.get("ok"):
                code = close_result.get("code", "?") if close_result else "?"
                msg = close_result.get("message", "") if close_result else ""
                print(f"  ❌ [LIVE] Close order FAILED for {pos.ticker}: "
                      f"code={code} {msg}")
                print(f"     Exchange={close_exchange} HoldID={pos.hold_id}")
                print(f"     Position remains open. Manual intervention required.")
                # DO NOT remove from self.positions — API close failed
                return trade

            print(f"  ✅ [LIVE] Close order accepted for {pos.ticker}: "
                  f"OrderID={close_result.get('order_id', '')} "
                  f"Exchange={close_exchange} HoldID={pos.hold_id}")

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
