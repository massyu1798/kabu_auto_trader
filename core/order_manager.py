"""発注・ポジション管理モジュール (v13.1: BT-aligned cooldown)"""

import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from core.api_client import KabuClient


@dataclass
class LivePosition:
    """保有ポジション"""
    ticker: str
    side: str              # "BUY" or "SELL"
    entry_price: float
    entry_time: datetime
    size: int
    stop_loss: float
    take_profit: float
    trailing_stop: float
    hold_id: str = ""      # kabu API の建玉ID
    order_id: str = ""     # 注文ID
    reason: str = ""
    session: str = ""      # "AM" or "PM" (for force close identification)


@dataclass
class LiveTrade:
    """決済済みトレード"""
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
    """発注・ポジション管理 (v13.1: BT-aligned cooldown)"""

    def __init__(self, client: KabuClient, config: dict):
        self.client = client
        self.paper_mode = config["mode"]["paper_trade"]
        self.trade_config = config["trade"]
        self.positions: list[LivePosition] = []
        self.trades: list[LiveTrade] = []
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.cooldown_until: dict[str, datetime] = {}

        # BT-aligned cooldown config (set by main_live after init)
        self.cooldown_config: dict = {
            "am_enabled": False,
            "am_loss_min": 75,   # default: 15 bars * 5min
            "am_win_min": 25,    # default: 5 bars * 5min
            "pm_enabled": False,
            "pm_loss_min": 30,   # default: 6 bars * 5min
            "pm_win_min": 10,    # default: 2 bars * 5min
        }

    def can_entry(self, ticker: str) -> bool:
        """エントリー可能か判定"""
        # 最大ポジション数チェック
        if len(self.positions) >= self.trade_config["max_positions"]:
            return False

        # 同銘柄で既にポジションあり
        if any(p.ticker == ticker for p in self.positions):
            return False

        # 日次損失上限チェック (BT-aligned: abs(daily_loss) < cap * max_daily_loss)
        max_loss = self.trade_config["initial_capital"] * self.trade_config["max_daily_loss"]
        if self.daily_pnl < 0 and abs(self.daily_pnl) >= max_loss:
            return False

        # クールダウン中
        if ticker in self.cooldown_until:
            if datetime.now() < self.cooldown_until[ticker]:
                return False

        return True

    def entry(self, ticker: str, side: str, price: float,
              size: int, stop_loss: float, take_profit: float,
              reason: str = "", session: str = "") -> bool:
        """新規エントリー"""

        now = datetime.now()

        if self.paper_mode:
            pos = LivePosition(
                ticker=ticker, side=side,
                entry_price=price, entry_time=now,
                size=size, stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=stop_loss,
                hold_id=f"PAPER_{ticker}_{int(time.time())}",
                reason=reason,
                session=session,
            )
            self.positions.append(pos)
            self.daily_trade_count += 1
            print(f"  📝 [PAPER] {side} {ticker} × {size}株 @ {price:.0f}円")
            print(f"       SL={stop_loss:.0f} TP={take_profit:.0f} | {reason}")
            return True
        else:
            margin_type = 1  # 制度信用
            result = self.client.send_margin_order(
                symbol=ticker,
                exchange=1,
                side=side,
                qty=size,
                order_type=1,  # 成行
                margin_trade_type=margin_type,
            )
            if result and result.get("OrderId"):
                pos = LivePosition(
                    ticker=ticker, side=side,
                    entry_price=price, entry_time=now,
                    size=size, stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop=stop_loss,
                    order_id=result["OrderId"],
                    reason=reason,
                    session=session,
                )
                self.positions.append(pos)
                self.daily_trade_count += 1
                print(f"  🔥 [LIVE] {side} {ticker} × {size}株 | OrderID={result['OrderId']}")
                return True
            else:
                print(f"  ❌ 発注失敗: {ticker}")
                return False

    def exit(self, pos: LivePosition, current_price: float, reason: str) -> LiveTrade:
        """ポジション決済"""

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
                exchange=1,
                side=close_side,
                qty=pos.size,
                hold_id=pos.hold_id,
                order_type=1,  # 成行
            )

        mode_tag = "PAPER" if self.paper_mode else "LIVE"
        pnl_str = f"+{pnl:,.0f}" if pnl >= 0 else f"{pnl:,.0f}"
        print(f"  {'✅' if pnl >= 0 else '❌'} [{mode_tag}] 決済 {pos.ticker} [{pos.session}] | {pnl_str}円 | {reason}")

        self.positions.remove(pos)
        self.trades.append(trade)
        self.daily_pnl += pnl

        # BT-aligned cooldown: use config-based minutes
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
        """日次サマリーを返す"""
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
