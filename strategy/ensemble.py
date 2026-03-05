"""��合スコアリングエンジン v6: 出来高分析追加 + v13 live wrapper"""

import pandas as pd
import yaml
from strategy.base import StrategyBase, Signal
from strategy.trend_follow import TrendFollow
from strategy.mean_reversion import MeanReversion
from strategy.breakout import Breakout
from strategy.volume_profile import VolumeProfile


# === 戦略レジストリ ===
STRATEGY_REGISTRY = {
    "trend_follow": TrendFollow,
    "mean_reversion": MeanReversion,
    "breakout": Breakout,
    "volume_profile": VolumeProfile,
}


class EnsembleEngine:
    """複数戦略を統合してシグナルを生成"""

    def __init__(self, config_path: str = "config/strategy_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.strategies: list[tuple[StrategyBase, float]] = []
        self._load_strategies()

    def _load_strategies(self):
        for name, cls in STRATEGY_REGISTRY.items():
            strat_config = self.config["strategies"].get(name, {})
            if strat_config.get("enabled", False):
                strategy = cls(strat_config.get("params", {}))
                weight = strat_config.get("weight", 1.0)
                self.strategies.append((strategy, weight))
                print(f"  ✅ {name} (weight={weight})")

    def generate_ensemble_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        ensemble_cfg = self.config["ensemble"]

        total_score = pd.Series(0.0, index=df.index)

        for strategy, weight in self.strategies:
            signals = strategy.generate_signals(df)
            scores = signals.apply(lambda s: s.score)
            reasons = signals.apply(lambda s: s.reason)

            result[f"{strategy.name}_score"] = scores
            result[f"{strategy.name}_reason"] = reasons
            total_score += scores * weight

        result["ensemble_score"] = total_score

        result["final_signal"] = "HOLD"
        result.loc[
            result["ensemble_score"] >= ensemble_cfg["buy_threshold"],
            "final_signal",
        ] = "BUY"
        result.loc[
            result["ensemble_score"] <= ensemble_cfg["sell_threshold"],
            "final_signal",
        ] = "SELL"

        return result

    # ================================================
    # v13: Live wrapper - evaluate last bar only
    # ================================================
    def evaluate_live(self, df: pd.DataFrame) -> tuple[str, float, str]:
        """
        Evaluate ensemble on a DataFrame of 5min bars and return
        the signal for the LAST bar only.

        Returns:
            (signal, score, detail_str)
            signal: "BUY" / "SELL" / "HOLD"
            score:  ensemble_score of last bar
            detail: human-readable breakdown
        """
        if df.empty or len(df) < 3:
            return ("HOLD", 0.0, "insufficient bars")

        result = self.generate_ensemble_signals(df)
        last = result.iloc[-1]
        signal = last["final_signal"]
        score = float(last["ensemble_score"])

        # Build detail string
        parts = []
        for strategy, weight in self.strategies:
            col = f"{strategy.name}_score"
            if col in result.columns:
                s = float(last[col])
                if s != 0:
                    parts.append(f"{strategy.name}={s:.1f}*{weight}")
        detail = " + ".join(parts) if parts else "no hits"

        return (signal, score, detail)
