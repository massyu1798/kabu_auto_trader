"""Order / position manager (v15.4: dynamic exchange from board)"""

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
    exchange: int = 1      # board-derived exchange code
    hold_id: str = ""      # kabu API hold ID
    order_id: str = ""     # order ID
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
    """Order / position manager (v15.4: dynamic exchange)"""

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

    def entry(self, ticker: str, side: str, price: float,
              size: int, stop_loss: float, take_profit: float,
              reason: str = "", session: str = "",
              exchange: int = 1) -> dict:
        """New entry.

        Args:
            exchange: board-derived exchange code (NOT hardcoded).

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
                hold_id=f"PAPER_{ticker}_{int(time.time())}",
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
                exchange=exchange,    # board-derived, not hardcoded
                side=side,
                qty=size,
                order_type=1,  # market
                margin_trade_type=mtt,
            )

            if result and result.get("ok"):
                pos = LivePosition(
                    ticker=ticker, side=side,
                    entry_price=price, entry_time=now,
                    size=size, stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop=stop_loss,
                    exchange=exchange,
                    order_id=result.get("order_id", ""),
                    reason=reason,
                    session=session,
                )
                self.positions.append(pos)
                self.daily_trade_count += 1
                print(f"  🔥 [LIVE] {side} {ticker} × {size}株 "
                      f"Exchange={exchange} MTT={mtt}({mtt_label}) "
                      f"| OrderID={result.get('order_id','')}")
                return result
            else:
                code = result.get('code', 0) if result else '?'
                print(f"  ❌ 発注失敗: {ticker} code={code} "
                      f"Exchange={exchange} MTT={mtt}({mtt_label})")
                if result:
                    return result
                return {"ok": False, "http": 0, "code": 0, "message": "no result from api"}

    def exit(self, pos: LivePosition, current_price: float, reason: str) -> LiveTrade:
        """Close position"""

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

        if not self.paper_mode and pos.hold_id and not pos.hold_id.startswith("PAPER"):
            close_side = "SELL" if pos.side == "BUY" else "BUY"
            self.client.send_margin_close(
                symbol=pos.ticker,
                exchange=pos.exchange,   # use stored exchange
                side=close_side,
                qty=pos.size,
                hold_id=pos.hold_id,
                order_type=1,
                margin_trade_type=self.margin_trade_type,
            )

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
