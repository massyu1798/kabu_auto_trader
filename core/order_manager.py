"""発注・ポジション管理モジュール"""

import time
from datetime import datetime
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


class OrderManager:
    """発注・ポジション管理"""

    def __init__(self, client: KabuClient, config: dict):
        self.client = client
        self.paper_mode = config["mode"]["paper_trade"]
        self.trade_config = config["trade"]
        self.positions: list[LivePosition] = []
        self.trades: list[LiveTrade] = []
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.cooldown_until: dict[str, datetime] = {}

    def can_entry(self, ticker: str) -> bool:
        """エントリー可能か判定"""
        # 最大ポジション数チェック
        if len(self.positions) >= self.trade_config["max_positions"]:
            return False

        # 同銘柄で既にポジションあり
        if any(p.ticker == ticker for p in self.positions):
            return False

        # 日次損失上限チェック
        max_loss = self.trade_config["initial_capital"] * self.trade_config["max_daily_loss"]
        if abs(self.daily_pnl) >= max_loss and self.daily_pnl < 0:
            return False

        # クールダウン中
        if ticker in self.cooldown_until:
            if datetime.now() < self.cooldown_until[ticker]:
                return False

        return True

    def entry(self, ticker: str, side: str, price: float,
              size: int, stop_loss: float, take_profit: float,
              reason: str = "") -> bool:
        """新規エントリー"""

        now = datetime.now()

        if self.paper_mode:
            # ペーパーモード: 発注せずにログのみ
            pos = LivePosition(
                ticker=ticker, side=side,
                entry_price=price, entry_time=now,
                size=size, stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=stop_loss,
                hold_id=f"PAPER_{ticker}_{int(time.time())}",
                reason=reason,
            )
            self.positions.append(pos)
            self.daily_trade_count += 1
            print(f"  📝 [PAPER] {side} {ticker} × {size}株 @ {price:.0f}円")
            print(f"       SL={stop_loss:.0f} TP={take_profit:.0f} | {reason}")
            return True
        else:
            # 本番モード: kabu API で発注
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
        )

        if not self.paper_mode and pos.hold_id and not pos.hold_id.startswith("PAPER"):
            # 本番モード: 返済注文
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
        print(f"  {'✅' if pnl >= 0 else '❌'} [{mode_tag}] 決済 {pos.ticker} | {pnl_str}円 | {reason}")

        self.positions.remove(pos)
        self.trades.append(trade)
        self.daily_pnl += pnl

        # クールダウン設定
        if pnl < 0:
            from datetime import timedelta
            self.cooldown_until[pos.ticker] = now + timedelta(minutes=90)

        return trade

    def get_daily_summary(self) -> str:
        """日次サマリーを返す"""
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self.trades)

        summary = f"""
日次サマリー
  トレード数: {len(self.trades)}
  勝ち: {len(wins)} / 負け: {len(losses)}
  損益: {total_pnl:+,.0f}円
  残ポジション: {len(self.positions)}
"""
        return summary