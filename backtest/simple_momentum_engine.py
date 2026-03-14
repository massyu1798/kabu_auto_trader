"""
シンプル順張り戦略専用バックテストエンジン

エントリー: 11:25バー close + スリッページ
エグジット: 12:30バー open（フォールバック: 12:35バー close）
SL/TP/タイムストップ: なし
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Optional

import pandas as pd
import yaml

# Side, PairTrade, PairBacktestResult は pair_meanrev_engine.py から流用（重複定義しない）
try:
    from backtest.pair_meanrev_engine import Side, PairTrade, PairBacktestResult
    from strategy.simple_momentum import SimpleMomentumEngine
    from strategy.pair_mean_reversion import UNIVERSE
except ImportError:
    from pair_meanrev_engine import Side, PairTrade, PairBacktestResult  # type: ignore
    from simple_momentum import SimpleMomentumEngine  # type: ignore
    from pair_mean_reversion import UNIVERSE  # type: ignore

logger = logging.getLogger(__name__)


class SimpleMomentumBacktestEngine:
    """シンプル順張り戦略専用バックテストエンジン。

    使用方法:
        engine = SimpleMomentumBacktestEngine("config/simple_momentum_config.yaml")
        result = engine.run(intraday_data, daily_data)
        print(engine.generate_report(result))
    """

    def __init__(self, config_path: str = "config/simple_momentum_config.yaml") -> None:
        """設定ファイルを読み込み、パラメータと戦略エンジンを初期化する。

        Args:
            config_path: simple_momentum_config.yaml のパス
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        g = self.config["global"]
        self.initial_capital: float = float(g["initial_capital"])
        self.max_positions_per_side: int = int(g["max_positions_per_side"])
        self.risk_per_position: float = float(g["risk_per_position"])
        self.commission_rate: float = float(g["commission_rate"])
        self.slippage_rate: float = float(g["slippage_rate"])

        r = self.config.get("risk", {})
        self.max_daily_loss: float = float(r.get("max_daily_loss", 0.015))

        self.strategy = SimpleMomentumEngine(config_path)

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    @staticmethod
    def _to_jst(df: pd.DataFrame) -> pd.DataFrame:
        """DataFrameのインデックスをAsia/Tokyoに変換する。"""
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
        """日足データから前日終値を取得する（ルックアヘッドバイアス回避）。"""
        if daily_df is None or daily_df.empty:
            return None
        try:
            trade_day = trade_date.date()
            idx_dates = [
                d.date() if hasattr(d, "date") else d for d in daily_df.index
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

    def _calc_pnl(
        self, entry: float, exit_p: float, size: int, side: Side
    ) -> tuple[float, float]:
        """PnLとPnL%を計算する（手数料込み）。

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
    ) -> PairBacktestResult:
        """バックテストを実行する。

        Args:
            intraday_data: 銘柄→5分足DataFrameの辞書（全期間）
            daily_data:    銘柄→日足DataFrameの辞書

        Returns:
            PairBacktestResult
        """
        capital = self.initial_capital
        all_trades: list[PairTrade] = []
        equity_curve: list[float] = []
        equity_dates: list = []
        daily_pnl: dict[str, float] = {}

        # ユニークな取引日を収集（最初の銘柄データを基準に）
        ref_df = self._to_jst(next(iter(intraday_data.values())).copy())
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

        prev_daily_loss = 0.0
        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

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
            # 前日終値（将来の拡張用）
            # ----------------------------------------------------------
            prev_close: dict[str, float] = {}
            for ticker in UNIVERSE:
                daily_df = daily_data.get(ticker)
                if daily_df is not None and not daily_df.empty:
                    pc = self._get_prev_close(daily_df, trade_date)
                    if pc is not None:
                        prev_close[ticker] = pc

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
            signal = self.strategy.generate_daily_signal(morning_data, prev_close)

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

                # エントリー価格: 11:25バー close にスリッページを適用
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

                # エグジット: 12:30バー open（後場寄り成行）
                # 12:30バーが取得できない場合のみ 12:35バーにフォールバック
                exit_raw: Optional[float] = None
                exit_reason = ""
                exit_ts: pd.Timestamp = entry_ts

                bar_1230 = self._get_bar_at(df, trade_date, 12, 30)
                if bar_1230 is not None:
                    exit_raw = float(bar_1230["open"])
                    exit_reason = "後場寄り決済"
                    exit_ts = bar_1230.name
                else:
                    # フォールバック: 12:35バー close
                    bar_1235 = self._get_bar_at(df, trade_date, 12, 35)
                    if bar_1235 is not None:
                        exit_raw = float(bar_1235["close"])
                        exit_reason = "後場遅延決済(12:35)"
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
                    entry_reason=f"{side.value} simple-momentum",
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
            # 日次PnL集計・equity更新
            # ----------------------------------------------------------
            day_pnl = sum(t.pnl for t in day_trades)
            capital += day_pnl
            daily_pnl[date_str] = day_pnl
            prev_daily_loss = day_pnl if day_pnl < 0 else 0.0

            if day_pnl < 0:
                current_loss_streak += 1
                current_win_streak = 0
                if current_loss_streak > max_loss_streak:
                    max_loss_streak = current_loss_streak
            else:
                current_win_streak += 1
                current_loss_streak = 0
                if current_win_streak > max_win_streak:
                    max_win_streak = current_win_streak

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
            max_win_streak=max_win_streak,
            max_loss_streak=max_loss_streak,
        )

    # ------------------------------------------------------------------
    # レポート生成
    # ------------------------------------------------------------------

    def generate_report(
        self, result: PairBacktestResult, initial_capital: Optional[float] = None
    ) -> str:
        """バックテスト結果の詳細テキストレポートを生成する。

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
        max_win = max((t.pnl for t in trades), default=0.0)
        max_loss = min((t.pnl for t in trades), default=0.0)
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

        # 最大ドローダウン
        eq = result.equity_curve
        max_dd_pct = 0.0
        max_dd_amount = 0.0
        if eq:
            peak = eq[0]
            for e in eq:
                if e > peak:
                    peak = e
                dd_amount = peak - e
                dd_pct = dd_amount / peak * 100.0 if peak > 0 else 0.0
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
                    max_dd_amount = dd_amount

        # 日別損益統計
        daily_vals = list(result.daily_pnl.values())
        sharpe = 0.0
        daily_mean = 0.0
        daily_std = 0.0
        daily_max = 0.0
        daily_min = 0.0
        if daily_vals:
            daily_mean = statistics.mean(daily_vals)
            daily_max = max(daily_vals)
            daily_min = min(daily_vals)
            if len(daily_vals) >= 2:
                daily_std = statistics.stdev(daily_vals)
                if daily_std > 0:
                    sharpe = (daily_mean / daily_std) * (252 ** 0.5)

        calmar = (total_ret / max_dd_pct) if max_dd_pct > 0 else float("inf")

        # LONG / SHORT 内訳
        long_trades = [t for t in trades if t.side == Side.LONG]
        short_trades = [t for t in trades if t.side == Side.SHORT]
        long_wins = [t for t in long_trades if t.pnl > 0]
        short_wins = [t for t in short_trades if t.pnl > 0]
        long_pnl = sum(t.pnl for t in long_trades)
        short_pnl = sum(t.pnl for t in short_trades)
        long_wr = len(long_wins) / len(long_trades) * 100.0 if long_trades else 0.0
        short_wr = len(short_wins) / len(short_trades) * 100.0 if short_trades else 0.0

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

        # 月別 PnL
        monthly_pnl: dict[str, float] = {}
        for t in trades:
            month_key = t.trade_date[:7]  # "YYYY-MM"
            monthly_pnl[month_key] = monthly_pnl.get(month_key, 0.0) + t.pnl
        monthly_lines = "\n".join(
            f"    {month}  {v:>+10,.0f}円"
            for month, v in sorted(monthly_pnl.items())
        )

        # 日別損益（上位/下位 5日）
        dpnl_items = sorted(result.daily_pnl.items(), key=lambda x: x[1])
        best5 = list(reversed(dpnl_items[-5:])) if dpnl_items else []
        worst5 = dpnl_items[:5] if dpnl_items else []
        best_lines = "\n".join(f"    {d}  {v:>+10,.0f}円" for d, v in best5)
        worst_lines = "\n".join(f"    {d}  {v:>+10,.0f}円" for d, v in worst5)

        # Exit 理由内訳
        exit_reasons: dict[str, int] = {}
        for t in trades:
            key = t.exit_reason.split("(")[0]
            exit_reasons[key] = exit_reasons.get(key, 0) + 1
        exit_lines = "\n".join(
            f"    {r}: {c}件"
            for r, c in sorted(exit_reasons.items(), key=lambda x: -x[1])
        )

        direction_str = "順張り" if self.strategy.direction == "momentum" else "逆張り"
        min_move = self.strategy.min_move_pct

        report = f"""
============================================================
     シンプル{direction_str}戦略 バックテストレポート
     (値幅 ≥ {min_move:.1f}%  後場寄り成行決済)
============================================================

■ 概要
  初期資金:          {ic:>15,.0f} 円
  最終資産:          {equity_final:>15,.0f} 円
  純損益:            {total_pnl:>+15,.0f} 円 ({total_ret:+.2f}%)
  最大DD:            {max_dd_pct:>14.2f} %  ({max_dd_amount:>+12,.0f} 円)
  Sharpe 比率:       {sharpe:>14.3f}
  Calmar 比率:       {calmar:>14.3f}

■ トレード統計
  総トレード数:      {total:>5d}
  勝ちトレード:      {len(wins):>5d} ({win_rate:.1f}%)
  負けトレード:      {len(losses):>5d} ({100 - win_rate:.1f}%)
  平均利益:          {avg_win:>+15,.0f} 円
  平均損失:          {avg_loss:>+15,.0f} 円
  最大利益:          {max_win:>+15,.0f} 円
  最大損失:          {max_loss:>+15,.0f} 円
  PF:                {pf:.3f}

■ 日別損益統計
  平均:              {daily_mean:>+15,.0f} 円
  標準偏差:          {daily_std:>15,.0f} 円
  最大:              {daily_max:>+15,.0f} 円
  最小:              {daily_min:>+15,.0f} 円

■ 連勝/連敗記録
  最大連勝日数:      {result.max_win_streak:>5d} 日
  最大連敗日数:      {result.max_loss_streak:>5d} 日

■ LONG / SHORT 内訳
  LONG:   {len(long_trades):>4d}件  勝ち{len(long_wins):>3d}件  勝率{long_wr:5.1f}%  PnL={long_pnl:>+12,.0f} 円
  SHORT:  {len(short_trades):>4d}件  勝ち{len(short_wins):>3d}件  勝率{short_wr:5.1f}%  PnL={short_pnl:>+12,.0f} 円

■ セクター別成績
{chr(10).join(sector_lines) if sector_lines else "  データなし"}

■ 月別 PnL
{monthly_lines if monthly_lines else "  データなし"}

■ 日別損益 Top 5
{best_lines if best_lines else "  データなし"}

■ 日別損益 Worst 5
{worst_lines if worst_lines else "  データなし"}

■ Exit 理由内訳
{exit_lines if exit_lines else "  データなし"}
============================================================
"""
        return report
