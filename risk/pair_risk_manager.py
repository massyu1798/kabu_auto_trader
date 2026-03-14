"""
ペア戦略リスク管理モジュール

機能:
  - ネットエクスポージャ計算
  - セクター集中度チェック
  - 日次損失チェック
  - β調整計算（日足データから過去60日βを算出）
  - 異常時停止判定（TOPIX先物急変、連続損失など）

将来のライブ取引（main_live.py + core/api_client.py）との接続を見据えて設計。
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ネットエクスポージャ計算
# ---------------------------------------------------------------------------


def calc_net_exposure(
    long_notional: dict[str, float],
    short_notional: dict[str, float],
) -> float:
    """ロング金額とショート金額からネットエクスポージャを計算する。

    ネットエクスポージャ = Σ(ロング金額) - Σ(ショート金額)

    Args:
        long_notional:  銘柄→ロング想定元本の辞書（円）
        short_notional: 銘柄→ショート想定元本の辞書（円）

    Returns:
        ネットエクスポージャ（円、正=ロング超過、負=ショート超過）
    """
    total_long = sum(long_notional.values())
    total_short = sum(short_notional.values())
    return total_long - total_short


def calc_gross_exposure(
    long_notional: dict[str, float],
    short_notional: dict[str, float],
) -> float:
    """グロスエクスポージャ（ロング + ショート合計）を計算する。

    Args:
        long_notional:  銘柄→ロング想定元本の辞書（円）
        short_notional: 銘柄→ショート想定元本の辞書（円）

    Returns:
        グロスエクスポージャ（円）
    """
    return sum(long_notional.values()) + sum(short_notional.values())


def check_net_exposure(
    long_notional: dict[str, float],
    short_notional: dict[str, float],
    capital: float,
    max_net_pct: float = 0.05,
) -> bool:
    """ネットエクスポージャが許容範囲内かチェックする。

    Args:
        long_notional:  銘柄→ロング想定元本の辞書（円）
        short_notional: 銘柄→ショート想定元本の辞書（円）
        capital:        現在の資本（円）
        max_net_pct:    ネットエクスポージャの最大許容比率（デフォルト 5%）

    Returns:
        True = 許容範囲内、False = 制限超過
    """
    net_exp = abs(calc_net_exposure(long_notional, short_notional))
    limit = capital * max_net_pct
    if net_exp > limit:
        logger.warning(
            f"ネットエクスポージャ制限超過: "
            f"{net_exp:,.0f}円 > {limit:,.0f}円 ({max_net_pct * 100:.1f}%)"
        )
        return False
    return True


# ---------------------------------------------------------------------------
# セクター集中度チェック
# ---------------------------------------------------------------------------


def check_sector_concentration(
    long_tickers: list[str],
    short_tickers: list[str],
    sector_map: dict[str, str],
    max_per_side: int = 1,
    no_cross_overlap: bool = True,
) -> tuple[bool, str]:
    """ロング・ショートのセクター集中度をチェックする。

    Args:
        long_tickers:    ロング銘柄リスト
        short_tickers:   ショート銘柄リスト
        sector_map:      銘柄→セクターのマッピング辞書
        max_per_side:    片側での同一セクター最大銘柄数（デフォルト 1）
        no_cross_overlap: True = ロング・ショート間での同セクター重複を禁止

    Returns:
        (OK: bool, 問題の説明: str)
    """
    # ロング側のセクター集計
    long_sectors: dict[str, int] = {}
    for t in long_tickers:
        sec = sector_map.get(t, "不明")
        long_sectors[sec] = long_sectors.get(sec, 0) + 1

    # ショート側のセクター集計
    short_sectors: dict[str, int] = {}
    for t in short_tickers:
        sec = sector_map.get(t, "不明")
        short_sectors[sec] = short_sectors.get(sec, 0) + 1

    # 片側制約チェック
    for sec, cnt in long_sectors.items():
        if cnt > max_per_side:
            msg = f"ロング側セクター集中超過: {sec} = {cnt}銘柄 (上限 {max_per_side})"
            logger.warning(msg)
            return False, msg

    for sec, cnt in short_sectors.items():
        if cnt > max_per_side:
            msg = f"ショート側セクター集中超過: {sec} = {cnt}銘柄 (上限 {max_per_side})"
            logger.warning(msg)
            return False, msg

    # ロング・ショート間の重複チェック
    if no_cross_overlap:
        overlap = set(long_sectors.keys()) & set(short_sectors.keys())
        if overlap:
            msg = f"ロング・ショート間でセクター重複: {overlap}"
            logger.warning(msg)
            return False, msg

    return True, "OK"


# ---------------------------------------------------------------------------
# 日次損失チェック
# ---------------------------------------------------------------------------


def check_daily_loss(
    current_daily_pnl: float,
    capital: float,
    max_loss_pct: float = 0.015,
) -> bool:
    """日次損失が許容範囲を超えているかチェックする。

    Args:
        current_daily_pnl: 当日累積 PnL（円、負 = 損失）
        capital:           現在の資本（円）
        max_loss_pct:      日次損失の最大許容比率（デフォルト 1.5%）

    Returns:
        True = 許容範囲内（エントリー可）、False = 損失超過（エントリー停止）
    """
    if current_daily_pnl >= 0:
        return True
    loss_pct = abs(current_daily_pnl) / capital
    if loss_pct > max_loss_pct:
        logger.warning(
            f"日次損失制限超過: {current_daily_pnl:+,.0f}円 "
            f"({loss_pct * 100:.2f}% > {max_loss_pct * 100:.1f}%) → 追加エントリー停止"
        )
        return False
    return True


# ---------------------------------------------------------------------------
# β調整計算
# ---------------------------------------------------------------------------


def calc_beta(
    stock_daily_df: pd.DataFrame,
    topix_daily_df: pd.DataFrame,
    trade_date: "pd.Timestamp",
    window: int = 60,
) -> float:
    """過去 window 日の日足データから市場β（対 TOPIX）を算出する。

    β = Cov(stock_returns, topix_returns) / Var(topix_returns)

    Args:
        stock_daily_df:  銘柄の日足 DataFrame（close 列必須）
        topix_daily_df:  TOPIX ETF の日足 DataFrame（close 列必須）
        trade_date:      取引日（この日より前のデータのみ使用、ルックアヘッドバイアス回避）
        window:          回帰ウィンドウ（デフォルト 60 日）

    Returns:
        β値（データ不足時は 1.0）
    """
    try:
        trade_day = trade_date.date() if hasattr(trade_date, "date") else trade_date

        def _get_returns(df: pd.DataFrame) -> list[float]:
            idx_dates = [d.date() if hasattr(d, "date") else d for d in df.index]
            prev_dates = sorted(d for d in idx_dates if d < trade_day)
            if len(prev_dates) < window + 1:
                return []
            recent = prev_dates[-(window + 1):]
            positions = [idx_dates.index(d) for d in recent]
            closes = df["close"].iloc[positions].values
            return [
                (closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, len(closes))
            ]

        stock_rets = _get_returns(stock_daily_df)
        topix_rets = _get_returns(topix_daily_df)
        n = min(len(stock_rets), len(topix_rets))
        if n < 10:
            return 1.0

        s = stock_rets[-n:]
        t = topix_rets[-n:]
        s_mean = sum(s) / n
        t_mean = sum(t) / n
        cov = sum((s[i] - s_mean) * (t[i] - t_mean) for i in range(n)) / n
        var_t = sum((t[i] - t_mean) ** 2 for i in range(n)) / n
        return float(cov / var_t) if var_t > 0 else 1.0

    except Exception as exc:
        logger.debug(f"calc_beta 失敗: {exc}")
        return 1.0


def calc_beta_adjusted_size(
    base_size: int,
    beta: float,
    target_beta: float = 1.0,
) -> int:
    """β調整後のポジションサイズを計算する。

    β=2.0 の銘柄は市場変動が 2 倍なので、β中立化のためには
    base_size / 2.0 に調整する。

    Args:
        base_size:    基本ポジションサイズ（株数）
        beta:         銘柄のβ値
        target_beta:  目標β（デフォルト 1.0 = 市場中立）

    Returns:
        β調整後のポジションサイズ（株数、最低 1）
    """
    if beta <= 0 or target_beta <= 0:
        return base_size
    adjusted = int(base_size * target_beta / beta)
    return max(1, adjusted)


def calc_portfolio_beta(
    tickers: list[str],
    sides: list[str],
    notionals: list[float],
    beta_dict: dict[str, float],
) -> float:
    """ポートフォリオ全体のβ加重平均を計算する。

    Args:
        tickers:    銘柄リスト
        sides:      ["LONG", "SHORT", ...] のリスト
        notionals:  各銘柄の想定元本リスト（円）
        beta_dict:  銘柄→β値の辞書

    Returns:
        加重平均β（ショートはマイナスで計算）
    """
    total_notional = sum(notionals)
    if total_notional == 0:
        return 0.0

    beta_sum = 0.0
    for ticker, side, notional in zip(tickers, sides, notionals):
        beta = beta_dict.get(ticker, 1.0)
        sign = 1.0 if side == "LONG" else -1.0
        beta_sum += sign * beta * notional

    return beta_sum / total_notional


# ---------------------------------------------------------------------------
# 異常時停止判定
# ---------------------------------------------------------------------------


def check_anomaly_stop(
    topix_return: float,
    consecutive_losses: int,
    current_daily_pnl: float,
    capital: float,
    max_topix_move_pct: float = 2.0,
    max_consecutive_losses: int = 5,
    max_daily_loss_pct: float = 0.015,
) -> tuple[bool, str]:
    """異常時停止判定をまとめて実行する。

    以下の条件のいずれかに該当する場合、当日の取引を停止する:
        1. TOPIX 前場リターンが ±max_topix_move_pct% 超
        2. consecutive_losses が max_consecutive_losses 以上
        3. 当日累積損失が資金の max_daily_loss_pct% 超

    Args:
        topix_return:         TOPIX 前場リターン（小数）
        consecutive_losses:   直近の連敗日数
        current_daily_pnl:    当日累積 PnL（円）
        capital:              現在の資本（円）
        max_topix_move_pct:   TOPIX フィルタ閾値（%）
        max_consecutive_losses: 連敗停止閾値
        max_daily_loss_pct:   日次損失停止閾値

    Returns:
        (停止すべき: bool, 理由: str)
    """
    # 1. TOPIX 地合いフィルタ
    if abs(topix_return) * 100.0 > max_topix_move_pct:
        reason = (
            f"TOPIX 急変: {topix_return * 100:.2f}% > ±{max_topix_move_pct}%"
        )
        logger.warning(f"異常停止判定: {reason}")
        return True, reason

    # 2. 連敗停止
    if consecutive_losses >= max_consecutive_losses:
        reason = f"連敗停止: {consecutive_losses}連敗 >= {max_consecutive_losses}"
        logger.warning(f"異常停止判定: {reason}")
        return True, reason

    # 3. 日次損失超過
    if not check_daily_loss(current_daily_pnl, capital, max_daily_loss_pct):
        reason = (
            f"日次損失超過: {current_daily_pnl:+,.0f}円 "
            f"({abs(current_daily_pnl) / capital * 100:.2f}% > {max_daily_loss_pct * 100:.1f}%)"
        )
        return True, reason

    return False, "OK"
