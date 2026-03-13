"""リスク管理システム v1.0

機能:
  - 日次/週間/月間損失トラッキング
  - セクター相関チェック（同一セクター2銘柄以上同時保有禁止）
  - ポートフォリオβ監視（β > 1.5 で警告）
  - 同時保有ポジション数管理（最大5銘柄）
  - 1銘柄最大建玉キャップ（有効資本の15%以下）
  - レバレッジ実効倍率監視（1.5倍以下目標）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# === セクターマッピング（日本株・主要銘柄） ===
SECTOR_MAP: dict[str, str] = {
    # 自動車
    "7203.T": "自動車", "7267.T": "自動車",
    # テクノロジー・半導体
    "6758.T": "テクノロジー", "8035.T": "テクノロジー",
    "6857.T": "テクノロジー", "6762.T": "テクノロジー",
    "6981.T": "テクノロジー", "6954.T": "テクノロジー",
    "6273.T": "テクノロジー",
    # 電機・精密
    "6501.T": "電機", "6503.T": "電機", "6702.T": "電機",
    "6752.T": "電機", "7751.T": "電機", "7741.T": "電機",
    "6861.T": "電機",
    # 通信
    "9432.T": "通信", "9433.T": "通信", "9984.T": "通信",
    # 金融・銀行
    "8306.T": "金融", "8316.T": "金融",
    # 保険
    "8766.T": "保険",
    # 商社
    "8058.T": "商社",
    # 小売
    "9983.T": "小売", "3382.T": "小売",
    # 製薬・ヘルスケア
    "4568.T": "製薬", "4502.T": "製薬", "4519.T": "製薬",
    # 化学
    "4063.T": "化学",
    # 鉄鋼
    "5401.T": "鉄鋼",
    # 機械
    "6367.T": "機械",
    # ETF（別カテゴリ）
    "1321.T": "ETF_日経225", "1306.T": "ETF_TOPIX",
    # 不動産
    "8801.T": "不動産",
    # 交通
    "9022.T": "交通",
    # エンタメ
    "4661.T": "エンタメ", "3659.T": "エンタメ", "7974.T": "エンタメ",
    # その他
    "6098.T": "その他", "4307.T": "その他", "2914.T": "その他",
}

# ベータ係数（簡易テーブル: vs 日経225）
BETA_MAP: dict[str, float] = {
    "7203.T": 1.1, "6758.T": 1.4, "9984.T": 1.6,
    "8306.T": 1.2, "8316.T": 1.1, "6857.T": 1.5,
    "4063.T": 1.0, "8035.T": 1.6, "6981.T": 1.3,
    "6954.T": 1.2, "7267.T": 1.1, "4568.T": 0.8,
    "6762.T": 1.3, "6501.T": 1.1, "6503.T": 1.0,
    "6702.T": 1.0, "6752.T": 1.2, "7751.T": 1.1,
    "9432.T": 0.7, "9433.T": 0.7, "8058.T": 1.0,
    "9983.T": 0.9, "3382.T": 0.7, "4502.T": 0.6,
    "4519.T": 0.7, "5401.T": 1.3, "8766.T": 0.9,
    "6367.T": 1.1, "8801.T": 0.9, "9022.T": 0.7,
    "4661.T": 0.8, "3659.T": 1.2, "7974.T": 1.1,
    "6098.T": 1.0, "4307.T": 0.9, "2914.T": 0.6,
    "7741.T": 1.0, "6273.T": 1.1, "6861.T": 1.3,
    "6981.T": 1.3, "1321.T": 1.0, "1306.T": 1.0,
}


@dataclass
class RiskLimits:
    """リスク管理パラメータ"""
    max_trade_loss_pct: float = 0.01        # 1トレード最大損失: 資金の1%
    max_daily_loss_pct: float = 0.025       # 1日最大損失: 資金の2.5%
    max_weekly_loss_pct: float = 0.05       # 週間最大損失: 資金の5%
    max_monthly_loss_pct: float = 0.10      # 月間最大損失: 資金の10%
    max_positions: int = 5                  # 同時保有ポジション上限
    max_position_pct: float = 0.15         # 1銘柄最大建玉: 信用枠の15%
    max_leverage: float = 1.5              # 実効レバレッジ目標上限
    max_portfolio_beta: float = 1.5        # ポートフォリオβ上限


@dataclass
class RiskState:
    """リスク状態トラッキング"""
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    daily_halted: bool = False
    weekly_halted: bool = False
    monthly_halted: bool = False
    daily_halt_count: int = 0
    weekly_halt_count: int = 0
    monthly_halt_count: int = 0
    current_day: object = None
    current_week: object = None
    current_month: object = None


@dataclass
class PortfolioStats:
    """ポートフォリオ統計"""
    total_trades: int = 0
    total_pnl: float = 0.0
    max_concurrent_positions: int = 0
    beta_history: list = field(default_factory=list)
    sector_violations: int = 0
    leverage_violations: int = 0


class RiskManager:
    """
    バックテスト内リスク管理マネージャー。

    使い方:
      rm = RiskManager(initial_capital=10_000_000)
      # 各バー・日次更新時
      rm.update_day(current_date)
      # エントリー可否チェック
      ok, reason = rm.check_entry(ticker, entry_value, current_positions)
      # トレードPnL記録
      rm.record_trade_pnl(pnl)
    """

    def __init__(
        self,
        initial_capital: float = 10_000_000,
        limits: Optional[RiskLimits] = None,
    ):
        self.capital = initial_capital
        self.limits = limits or RiskLimits()
        self.state = RiskState()
        self.stats = PortfolioStats()

    # ------------------------------------------------------------------
    # Day / Week / Month boundary
    # ------------------------------------------------------------------

    def update_day(self, day) -> None:
        """日付が変わったときに呼び出す"""
        day_val = day.date() if hasattr(day, "date") else day
        week_key = day_val.isocalendar()[:2] if hasattr(day_val, "isocalendar") else None
        month_key = (day_val.year, day_val.month) if hasattr(day_val, "year") else None

        if self.state.current_week != week_key:
            self.state.weekly_pnl = 0.0
            self.state.current_week = week_key
            self.state.weekly_halted = False

        if self.state.current_month != month_key:
            self.state.monthly_pnl = 0.0
            self.state.current_month = month_key
            self.state.monthly_halted = False

        if self.state.current_day != day_val:
            self.state.daily_pnl = 0.0
            self.state.current_day = day_val
            self.state.daily_halted = False

    # ------------------------------------------------------------------
    # Loss limit checks
    # ------------------------------------------------------------------

    def _check_daily_limit(self) -> bool:
        """日次損失限度チェック。超過していれば True（停止）"""
        exceeded = (
            abs(min(0.0, self.state.daily_pnl))
            >= self.capital * self.limits.max_daily_loss_pct
        )
        if exceeded and not self.state.daily_halted:
            self.state.daily_halted = True
            self.state.daily_halt_count += 1
        return self.state.daily_halted

    def _check_weekly_limit(self) -> bool:
        exceeded = (
            abs(min(0.0, self.state.weekly_pnl))
            >= self.capital * self.limits.max_weekly_loss_pct
        )
        if exceeded and not self.state.weekly_halted:
            self.state.weekly_halted = True
            self.state.weekly_halt_count += 1
        return self.state.weekly_halted

    def _check_monthly_limit(self) -> bool:
        exceeded = (
            abs(min(0.0, self.state.monthly_pnl))
            >= self.capital * self.limits.max_monthly_loss_pct
        )
        if exceeded and not self.state.monthly_halted:
            self.state.monthly_halted = True
            self.state.monthly_halt_count += 1
        return self.state.monthly_halted

    # ------------------------------------------------------------------
    # Entry check
    # ------------------------------------------------------------------

    def check_entry(
        self,
        ticker: str,
        entry_value: float,
        current_positions: list,
    ) -> tuple[bool, str]:
        """
        新規エントリーの可否を判定。
        戻り値: (ok: bool, reason: str)
        """
        # 損失停止チェック
        if self._check_daily_limit():
            return False, "日次損失上限到達"
        if self._check_weekly_limit():
            return False, "週間損失上限到達"
        if self._check_monthly_limit():
            return False, "月間損失上限到達"

        # 同時保有数チェック
        if len(current_positions) >= self.limits.max_positions:
            return False, f"最大ポジション数({self.limits.max_positions})到達"

        # 1銘柄最大建玉チェック
        max_by_cap = self.capital * self.limits.max_position_pct
        if entry_value > max_by_cap:
            return False, f"1銘柄最大建玉超過({entry_value:,.0f} > {max_by_cap:,.0f})"

        # セクター相関チェック（同一セクター2銘柄以上禁止）
        ticker_sector = SECTOR_MAP.get(ticker, "unknown")
        if ticker_sector not in ("unknown", "ETF_日経225", "ETF_TOPIX"):
            same_sector = [
                p for p in current_positions
                if SECTOR_MAP.get(getattr(p, "ticker", ""), "unknown") == ticker_sector
            ]
            if same_sector:
                self.stats.sector_violations += 1
                return False, f"セクター重複({ticker_sector})"

        # ETF重複チェック（日経225先物とETFの重複禁止）
        if ticker in ("1321.T", "1306.T"):
            etf_positions = [
                p for p in current_positions
                if getattr(p, "ticker", "") in ("1321.T", "1306.T")
            ]
            if etf_positions:
                return False, "ETF重複ポジション禁止"

        # ポートフォリオβチェック
        new_beta = self.calc_portfolio_beta(current_positions, ticker, entry_value)
        if new_beta > self.limits.max_portfolio_beta:
            return False, f"ポートフォリオβ超過({new_beta:.2f}>{self.limits.max_portfolio_beta})"

        return True, ""

    # ------------------------------------------------------------------
    # PnL recording
    # ------------------------------------------------------------------

    def record_trade_pnl(self, pnl: float) -> None:
        """トレードPnLを記録する"""
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl
        self.state.monthly_pnl += pnl
        self.stats.total_pnl += pnl
        self.stats.total_trades += 1

    # ------------------------------------------------------------------
    # Portfolio beta
    # ------------------------------------------------------------------

    def calc_portfolio_beta(
        self,
        current_positions: list,
        new_ticker: str = "",
        new_value: float = 0.0,
    ) -> float:
        """現在ポジション＋新規エントリー後のポートフォリオβを計算"""
        total_value = new_value
        weighted_beta = BETA_MAP.get(new_ticker, 1.0) * new_value
        for pos in current_positions:
            ticker = getattr(pos, "ticker", "")
            val = getattr(pos, "entry_price", 0) * getattr(pos, "size", 0)
            beta = BETA_MAP.get(ticker, 1.0)
            total_value += val
            weighted_beta += beta * val
        if total_value <= 0:
            return 0.0
        return weighted_beta / total_value

    def snapshot_beta(self, current_positions: list) -> float:
        """現在ポジションのβスナップショット"""
        beta = self.calc_portfolio_beta(current_positions)
        self.stats.beta_history.append(beta)
        if beta > self.limits.max_leverage:
            self.stats.leverage_violations += 1
        return beta

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def get_report(self) -> dict:
        """リスク管理レポートデータを返す"""
        avg_beta = (
            sum(self.stats.beta_history) / len(self.stats.beta_history)
            if self.stats.beta_history else 0.0
        )
        max_beta = max(self.stats.beta_history) if self.stats.beta_history else 0.0
        return {
            "daily_halt_count": self.state.daily_halt_count,
            "weekly_halt_count": self.state.weekly_halt_count,
            "monthly_halt_count": self.state.monthly_halt_count,
            "sector_violations": self.stats.sector_violations,
            "leverage_violations": self.stats.leverage_violations,
            "avg_portfolio_beta": avg_beta,
            "max_portfolio_beta": max_beta,
        }

    def format_report(self) -> str:
        """リスク管理レポートを文字列にフォーマット"""
        r = self.get_report()
        lines = [
            "\n■ リスク管理レポート",
            f"  日次最大損失到達回数:    {r['daily_halt_count']:>4} 回",
            f"  週間停止回数:            {r['weekly_halt_count']:>4} 回",
            f"  月間停止回数:            {r['monthly_halt_count']:>4} 回",
            f"  セクター重複拒否回数:    {r['sector_violations']:>4} 回",
            f"  ポートフォリオβ(平均):  {r['avg_portfolio_beta']:>7.2f}",
            f"  ポートフォリオβ(最大):  {r['max_portfolio_beta']:>7.2f}",
        ]
        return "\n".join(lines)
