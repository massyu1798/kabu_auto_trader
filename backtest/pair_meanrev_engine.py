"""
ペアモメンタム継続戦略専用バックテストエンジン

エントリー: 11:30 前場引け（11:25 バー close + スリッページ）
エグジット:
  immediate モード → 12:30 後場寄り（12:30 バー open + スリッページ）
  delayed   モード → 12:30〜12:35 で SL/TP 判定、12:35 タイムストップ
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd
import yaml

try:
    from strategy.pair_mean_reversion import PairMeanReversionEngine, UNIVERSE
except ImportError:
    from pair_mean_reversion import PairMeanReversionEngine, UNIVERSE  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# データクラス定義
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# エンジン本体
# ---------------------------------------------------------------------------


class PairMeanRevBacktestEngine:
    """ペアモメンタム継続戦略専用バックテストエンジン。

    使用方法:
        engine = PairMeanRevBacktestEngine("config/pair_meanrev_config.yaml")
        result = engine.run(intraday_data, daily_data, topix_intraday)
        print(engine.generate_report(result))
    """

    def __init__(self, config_path: str = "config/pair_meanrev_config.yaml") -> None:
        """設定ファイルを読み込み、パラメータと戦略エンジンを初期化する。

        Args:
            config_path: pair_meanrev_config.yaml のパス
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        g = self.config["global"]
        self.initial_capital: float = float(g["initial_capital"])
        self.max_positions_per_side: int = int(g["max_positions_per_side"])
        self.risk_per_position: float = float(g["risk_per_position"])
        self.commission_rate: float = float(g["commission_rate"])
        self.slippage_rate: float = float(g["slippage_rate"])

        t = self.config["timing"]
        self.exit_mode: str = str(t.get("exit_mode", "immediate"))

        e = self.config["exit"]
        self.stop_loss_pct: float = float(e["stop_loss_pct"])
        self.take_profit_pct: float = float(e["take_profit_pct"])

        r = self.config["risk"]
        self.max_daily_loss: float = float(r["max_daily_loss"])

        self.strategy = PairMeanReversionEngine(config_path)

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    @staticmethod
    def _to_jst(df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame のインデックスを Asia/Tokyo に変換する。"""
        if df is None or df.empty:
            return df
        if df.index.tz is None:
            df = df.copy()
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Tokyo")
        elif str(df.index.tz) != "Asia/Tokyo":
            df = df.copy()
            df.index = df.index.tz_convert("Asia/Tokyo")
        return df

    def _get_morning_bars(
        self, df: pd.DataFrame, trade_date: pd.Timestamp
    ) -> pd.DataFrame:
        """指定日の前場バー（9:00〜11:25）を取得する。"""
        df = self._to_jst(df)
        day = trade_date.date()
        day_bars = df[df.index.date == day]
        # 9:00〜11:30 の範囲から 11:25 以前のみ抽出
        morning = day_bars[
            (day_bars.index.hour >= 9)
            & (
                (day_bars.index.hour < 11)
                | ((day_bars.index.hour == 11) & (day_bars.index.minute <= 25))
            )
        ]
        return morning

    def _get_bar_at(
        self,
        df: pd.DataFrame,
        trade_date: pd.Timestamp,
        hour: int,
        minute: int,
    ) -> Optional[pd.Series]:
        """指定日・時刻（hour:minute）のバーを取得する。"""
        df = self._to_jst(df)
        day = trade_date.date()
        day_bars = df[df.index.date == day]
        mask = (day_bars.index.hour == hour) & (day_bars.index.minute == minute)
        bars = day_bars[mask]
        if bars.empty:
            return None
        return bars.iloc[0]

    def _get_prev_close(
        self, daily_df: pd.DataFrame, trade_date: pd.Timestamp
    ) -> Optional[float]:
        """日足データから前日終値を取得する。

        ルックアヘッドバイアスを防ぐため trade_date より前の最新日を参照する。
        """
        if daily_df.empty:
            return None
        try:
            trade_day = trade_date.date()
            idx = daily_df.index
            # date オブジェクトに統一
            idx_dates = [
                d.date() if hasattr(d, "date") else d for d in idx
            ]
            prev_dates = [d for d in idx_dates if d < trade_day]
            if not prev_dates:
                return None
            prev_day = max(prev_dates)
            pos = idx_dates.index(prev_day)
            return float(daily_df["close"].iloc[pos])
        except Exception as exc:
            logger.debug(f"_get_prev_close 失敗: {exc}")
            return None

    def _get_avg_volume(
        self, daily_df: pd.DataFrame, trade_date: pd.Timestamp, window: int = 20
    ) -> Optional[float]:
        """日足データから 20 日平均出来高を取得する（ルックアヘッドバイアス回避）。"""
        if daily_df.empty:
            return None
        try:
            trade_day = trade_date.date()
            idx_dates = [
                d.date() if hasattr(d, "date") else d for d in daily_df.index
            ]
            prev_dates = sorted(d for d in idx_dates if d < trade_day)
            if len(prev_dates) < 5:
                return None
            recent = prev_dates[-window:]
            positions = [idx_dates.index(d) for d in recent]
            vols = daily_df["volume"].iloc[positions]
            return float(vols.mean())
        except Exception as exc:
            logger.debug(f"_get_avg_volume 失敗: {exc}")
            return None

    def _entry_price(self, raw: float, side: Side) -> float:
        """エントリー価格を算出する（スリッページ適用）。"""
        if side == Side.LONG:
            return raw * (1.0 + self.slippage_rate)
        return raw * (1.0 - self.slippage_rate)

    def _exit_price(self, raw: float, side: Side) -> float:
        """エグジット価格を算出する（スリッページ適用）。"""
        if side == Side.LONG:
            return raw * (1.0 - self.slippage_rate)
        return raw * (1.0 + self.slippage_rate)

    def _check_sl_tp(
        self,
        bar: pd.Series,
        side: Side,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[tuple[float, str]]:
        """バーの high/low で SL・TP ヒットを判定する。

        Returns:
            (生の執行価格, 理由文字列) または None（ヒットなし）
        """
        high = float(bar["high"])
        low = float(bar["low"])

        if side == Side.LONG:
            if low <= stop_loss:
                return stop_loss, f"損切り({stop_loss:.0f})"
            if high >= take_profit:
                return take_profit, f"利確({take_profit:.0f})"
        else:  # SHORT
            if high >= stop_loss:
                return stop_loss, f"損切り({stop_loss:.0f})"
            if low <= take_profit:
                return take_profit, f"利確({take_profit:.0f})"
        return None

    def _calc_pnl(
        self, entry: float, exit_p: float, size: int, side: Side
    ) -> tuple[float, float]:
        """PnL と PnL% を計算する（手数料込み）。

        Returns:
            (pnl, pnl_pct)
        """
        gross = (exit_p - entry) * size if side == Side.LONG else (entry - exit_p) * size
        commission = (
            abs(entry * size * self.commission_rate)
            + abs(exit_p * size * self.commission_rate)
        )
        pnl = gross - commission
        notional = entry * size
        pnl_pct = (pnl / notional * 100.0) if notional > 0 else 0.0
        return pnl, pnl_pct

    # ------------------------------------------------------------------
    # メイン実行
    # ------------------------------------------------------------------

    def run(
        self,
        intraday_data: dict[str, pd.DataFrame],
        daily_data: dict[str, pd.DataFrame],
        topix_intraday: pd.DataFrame,
        topix_daily: Optional[pd.DataFrame] = None,
    ) -> PairBacktestResult:
        """バックテストを実行する。

        Args:
            intraday_data:   銘柄→5 分足 DataFrame の辞書（全期間）
            daily_data:      銘柄→日足 DataFrame の辞書
            topix_intraday:  TOPIX ETF（1306.T）の 5 分足
            topix_daily:     TOPIX ETF の日足（省略可）

        Returns:
            PairBacktestResult
        """
        capital = self.initial_capital
        all_trades: list[PairTrade] = []
        equity_curve: list[float] = []
        equity_dates: list = []
        daily_pnl: dict[str, float] = {}

        # タイムゾーン統一
        topix_intraday = self._to_jst(topix_intraday.copy())
        if topix_daily is not None:
            topix_daily = self._to_jst(topix_daily.copy())

        # ユニークな取引日を収集（TOPIX を基準にする）
        if topix_intraday.empty:
            # フォールバック: 最初の銘柄データを基準に
            ref_df = self._to_jst(next(iter(intraday_data.values())).copy())
        else:
            ref_df = topix_intraday

        trading_dates = sorted(
            {pd.Timestamp(ts.date()) for ts in ref_df.index}
        )

        if not trading_dates:
            logger.warning("取引日が見つかりません")
            return PairBacktestResult()

        logger.info(
            f"バックテスト期間: {trading_dates[0].date()} "
            f"〜 {trading_dates[-1].date()} ({len(trading_dates)} 日)"
        )

        prev_daily_loss = 0.0  # 直前の日次損失（停止判定用）

        for trade_date in trading_dates:
            date_str = str(trade_date.date())

            # ----------------------------------------------------------
            # 前場データ収集
            # ----------------------------------------------------------
            morning_data: dict[str, pd.DataFrame] = {}
            for ticker, df in intraday_data.items():
                m = self._get_morning_bars(df, trade_date)
                if not m.empty:
                    morning_data[ticker] = m

            if not morning_data:
                continue

            # ----------------------------------------------------------
            # 前日終値・20 日平均出来高
            # ----------------------------------------------------------
            prev_close: dict[str, float] = {}
            avg_volume: dict[str, float] = {}
            all_tickers = list(UNIVERSE.keys()) + ["1306.T"]

            for ticker in all_tickers:
                daily_df = daily_data.get(ticker)
                if daily_df is None and ticker == "1306.T":
                    daily_df = topix_daily

                if daily_df is not None and not daily_df.empty:
                    pc = self._get_prev_close(daily_df, trade_date)
                    if pc is not None:
                        prev_close[ticker] = pc
                    av = self._get_avg_volume(daily_df, trade_date)
                    if av is not None:
                        avg_volume[ticker] = av

            # ----------------------------------------------------------
            # TOPIX 前場データ
            # ----------------------------------------------------------
            topix_morning = self._get_morning_bars(topix_intraday, trade_date)

            # ----------------------------------------------------------
            # 日次損失上限チェック（前日ベース）
            # ----------------------------------------------------------
            if abs(prev_daily_loss) > capital * self.max_daily_loss:
                logger.info(
                    f"[{date_str}] 日次損失リミット到達 "
                    f"({prev_daily_loss:+,.0f}円) → スキップ"
                )
                equity_curve.append(capital)
                equity_dates.append(trade_date)
                prev_daily_loss = 0.0
                continue

            # ----------------------------------------------------------
            # シグナル生成
            # ----------------------------------------------------------
            signal = self.strategy.generate_daily_signal(
                morning_data, prev_close, topix_morning, avg_volume
            )

            if signal is None:
                logger.info(f"[{date_str}] シグナルなし")
                equity_curve.append(capital)
                equity_dates.append(trade_date)
                continue

            long_tickers, short_tickers = signal
            logger.info(
                f"[{date_str}] シグナル → LONG={long_tickers}, SHORT={short_tickers}"
            )

            # ----------------------------------------------------------
            # エントリー・エグジット
            # ----------------------------------------------------------
            day_trades: list[PairTrade] = []
            selected = (
                [(t, Side.LONG) for t in long_tickers]
                + [(t, Side.SHORT) for t in short_tickers]
            )

            for ticker, side in selected:
                df = intraday_data.get(ticker)
                if df is None:
                    continue
                df = self._to_jst(df)

                # エントリー価格: 11:25 バー close を前場引け価格の代理とする
                entry_bar = self._get_bar_at(df, trade_date, 11, 25)
                if entry_bar is None:
                    # フォールバック: 前場最後のバー
                    morning = self._get_morning_bars(df, trade_date)
                    if morning.empty:
                        logger.debug(f"[{date_str}] {ticker}: 前場データなし")
                        continue
                    entry_bar = morning.iloc[-1]

                entry_raw = float(entry_bar["close"])
                ep = self._entry_price(entry_raw, side)
                entry_ts: pd.Timestamp = entry_bar.name

                # ポジションサイズ（資金 × risk_per_position / エントリー価格）
                position_value = capital * self.risk_per_position
                size = math.floor(position_value / ep)
                if size <= 0:
                    continue

                # SL / TP 価格
                if side == Side.LONG:
                    sl = ep * (1.0 - self.stop_loss_pct)
                    tp = ep * (1.0 + self.take_profit_pct)
                else:
                    sl = ep * (1.0 + self.stop_loss_pct)
                    tp = ep * (1.0 - self.take_profit_pct)

                # エグジット判定
                exit_raw: Optional[float] = None
                exit_reason = ""
                exit_ts: pd.Timestamp = entry_ts

                if self.exit_mode == "immediate":
                    # 後場寄り（12:30 バー open）で即時決済
                    bar_1230 = self._get_bar_at(df, trade_date, 12, 30)
                    if bar_1230 is not None:
                        exit_raw = float(bar_1230["open"])
                        exit_reason = "後場寄り決済"
                        exit_ts = bar_1230.name
                    else:
                        # フォールバック: 12:35 バー close
                        bar_1235 = self._get_bar_at(df, trade_date, 12, 35)
                        if bar_1235 is not None:
                            exit_raw = float(bar_1235["close"])
                            exit_reason = "後場遅延決済(12:35)"
                            exit_ts = bar_1235.name

                else:  # delayed モード
                    # 12:30 バーで SL/TP チェック
                    bar_1230 = self._get_bar_at(df, trade_date, 12, 30)
                    if bar_1230 is not None:
                        hit = self._check_sl_tp(bar_1230, side, sl, tp)
                        if hit:
                            exit_raw, exit_reason = hit
                            exit_ts = bar_1230.name

                    # 未決済なら 12:35 バーで SL/TP or タイムストップ
                    if exit_raw is None:
                        bar_1235 = self._get_bar_at(df, trade_date, 12, 35)
                        if bar_1235 is not None:
                            hit = self._check_sl_tp(bar_1235, side, sl, tp)
                            if hit:
                                exit_raw, exit_reason = hit
                            else:
                                exit_raw = float(bar_1235["close"])
                                exit_reason = "タイムストップ(12:35)"
                            exit_ts = bar_1235.name

                if exit_raw is None:
                    # 後場データ取得不可: エントリー価格で flat
                    exit_raw = entry_raw
                    exit_reason = "データ不足(flat)"
                    logger.warning(f"[{date_str}] {ticker}: 後場データなし → flat 決済")

                xp = self._exit_price(exit_raw, side)
                pnl, pnl_pct = self._calc_pnl(ep, xp, size, side)
                sector = UNIVERSE.get(ticker, {}).get("sector", "")

                trade = PairTrade(
                    pair_id=date_str,
                    ticker=ticker,
                    side=side,
                    entry_price=ep,
                    exit_price=xp,
                    entry_date=entry_ts,
                    exit_date=exit_ts,
                    size=size,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    entry_reason=f"{side.value} momentum-based",
                    exit_reason=exit_reason,
                    sector=sector,
                    trade_date=date_str,
                )
                day_trades.append(trade)

                logger.info(
                    f"  {ticker:8s} [{side.value:5s}]"
                    f"  EP={ep:8.1f} XP={xp:8.1f}"
                    f"  size={size:5d}  PnL={pnl:+,.0f}円"
                    f"  ({exit_reason})"
                )

            # ----------------------------------------------------------
            # 日次 PnL 集計・equity 更新
            # ----------------------------------------------------------
            day_pnl = sum(t.pnl for t in day_trades)
            capital += day_pnl
            daily_pnl[date_str] = day_pnl
            prev_daily_loss = day_pnl if day_pnl < 0 else 0.0

            all_trades.extend(day_trades)
            equity_curve.append(capital)
            equity_dates.append(trade_date)

            logger.info(
                f"[{date_str}] 日次PnL={day_pnl:+,.0f}円  "
                f"資本={capital:,.0f}円  ({len(day_trades)} トレード)"
            )

        return PairBacktestResult(
            trades=all_trades,
            equity_curve=equity_curve,
            dates=equity_dates,
            daily_pnl=daily_pnl,
        )

    # ------------------------------------------------------------------
    # レポート生成
    # ------------------------------------------------------------------

    def generate_report(
        self, result: PairBacktestResult, initial_capital: Optional[float] = None
    ) -> str:
        """バックテスト結果のテキストレポートを生成する。

        通常の勝率・PF・最大 DD に加えて以下を含む:
            - LONG / SHORT 側別成績
            - セクター別成績
            - 日別損益サマリ（上位 / 下位 5 日）
            - Sharpe 比率（√252 スケール）
            - Calmar 比率

        Args:
            result:          run() の戻り値
            initial_capital: 初期資本（省略時は self.initial_capital）

        Returns:
            整形済みテキストレポート
        """
        ic = initial_capital if initial_capital is not None else self.initial_capital
        trades = result.trades

        if not trades:
            return "\n⚠️ トレードが1件も発生しませんでした。"

        total = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / total * 100.0

        total_pnl = sum(t.pnl for t in trades)
        total_ret = total_pnl / ic * 100.0
        equity_final = ic + total_pnl

        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

        # 最大ドローダウン
        eq = result.equity_curve
        max_dd_pct = 0.0
        if eq:
            peak = eq[0]
            for e in eq:
                if e > peak:
                    peak = e
                dd = (peak - e) / peak * 100.0 if peak > 0 else 0.0
                if dd > max_dd_pct:
                    max_dd_pct = dd

        # Sharpe 比率（日次 PnL ベース）
        daily_vals = list(result.daily_pnl.values())
        sharpe = 0.0
        if len(daily_vals) >= 2:
            mean_d = statistics.mean(daily_vals)
            std_d = statistics.stdev(daily_vals)
            if std_d > 0:
                sharpe = (mean_d / std_d) * (252 ** 0.5)

        # Calmar 比率
        calmar = (total_ret / max_dd_pct) if max_dd_pct > 0 else float("inf")

        # LONG / SHORT 内訳
        long_trades = [t for t in trades if t.side == Side.LONG]
        short_trades = [t for t in trades if t.side == Side.SHORT]
        long_wins = [t for t in long_trades if t.pnl > 0]
        short_wins = [t for t in short_trades if t.pnl > 0]
        long_pnl = sum(t.pnl for t in long_trades)
        short_pnl = sum(t.pnl for t in short_trades)

        # セクター別成績
        sector_stats: dict[str, dict] = {}
        for t in trades:
            s = t.sector or "不明"
            if s not in sector_stats:
                sector_stats[s] = {"pnl": 0.0, "count": 0, "wins": 0}
            sector_stats[s]["pnl"] += t.pnl
            sector_stats[s]["count"] += 1
            if t.pnl > 0:
                sector_stats[s]["wins"] += 1

        sector_lines = []
        for sec, st in sorted(sector_stats.items(), key=lambda x: -x[1]["pnl"]):
            wr = st["wins"] / st["count"] * 100.0 if st["count"] > 0 else 0.0
            sector_lines.append(
                f"    {sec:12s}  {st['pnl']:>+10,.0f}円  "
                f"{st['count']:3d}件  勝率{wr:.0f}%"
            )

        # 日別損益（上位 / 下位 5 日）
        dpnl_items = sorted(result.daily_pnl.items(), key=lambda x: x[1])
        best5 = list(reversed(dpnl_items[-5:])) if len(dpnl_items) >= 1 else []
        worst5 = dpnl_items[:5] if len(dpnl_items) >= 1 else []

        best_lines = "\n".join(
            f"    {d}  {v:>+10,.0f}円" for d, v in best5
        )
        worst_lines = "\n".join(
            f"    {d}  {v:>+10,.0f}円" for d, v in worst5
        )

        # Exit 理由内訳
        exit_reasons: dict[str, int] = {}
        for t in trades:
            key = t.exit_reason.split("(")[0]
            exit_reasons[key] = exit_reasons.get(key, 0) + 1
        exit_lines = "\n".join(
            f"    {r}: {c}件"
            for r, c in sorted(exit_reasons.items(), key=lambda x: -x[1])
        )

        report = f"""
============================================================
        ペアモメンタム継続戦略 バックテストレポート
============================================================

■ 概要
  初期資金:          {ic:>15,.0f} 円
  最終資産:          {equity_final:>15,.0f} 円
  純損益:            {total_pnl:>+15,.0f} 円 ({total_ret:+.2f}%)
  最大DD:            {max_dd_pct:>14.2f} %
  Sharpe 比率:       {sharpe:>14.3f}
  Calmar 比率:       {calmar:>14.3f}

■ トレード統計
  総トレード数:      {total:>5d}
  勝ちトレード:      {len(wins):>5d} ({win_rate:.1f}%)
  負けトレード:      {len(losses):>5d} ({100 - win_rate:.1f}%)
  平均利益:          {avg_win:>+15,.0f} 円
  平均損失:          {avg_loss:>+15,.0f} 円
  PF:                {pf:.3f}

■ LONG / SHORT 内訳
  LONG:   {len(long_trades):>4d}件  勝ち{len(long_wins):>3d}件  PnL={long_pnl:>+12,.0f} 円
  SHORT:  {len(short_trades):>4d}件  勝ち{len(short_wins):>3d}件  PnL={short_pnl:>+12,.0f} 円

■ セクター別成績
{chr(10).join(sector_lines) if sector_lines else "  データなし"}

■ 日別損益 Top 5
{best_lines if best_lines else "  データなし"}

■ 日別損益 Worst 5
{worst_lines if worst_lines else "  データなし"}

■ Exit 理由内訳
{exit_lines if exit_lines else "  データなし"}
============================================================
"""
        return report
