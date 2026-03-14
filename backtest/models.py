"""
共有データモデル定義

バックテストエンジン間で共有する Side, PairTrade, PairBacktestResult を定義する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Side(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class PairTrade:
    """ペアトレードの 1 件分の記録。

    pair_id は同一日に発生した全トレードで共通の日付文字列（例: "2025-01-15"）。
    """

    pair_id: str
    ticker: str
    side: Side
    entry_price: float
    exit_price: float
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    size: int
    pnl: float
    pnl_pct: float
    entry_reason: str = ""
    exit_reason: str = ""
    sector: str = ""
    trade_date: str = ""


@dataclass
class PairBacktestResult:
    """バックテスト結果の集約コンテナ。"""

    trades: list[PairTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    dates: list = field(default_factory=list)
    daily_pnl: dict[str, float] = field(default_factory=dict)
    # 連勝/連敗記録
    max_win_streak: int = 0   # 最大連勝日数
    max_loss_streak: int = 0  # 最大連敗日数
