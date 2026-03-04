"""
パラメータ最適化ツール (修正版)
"""

import yaml
import pandas as pd
from backtest.engine import BacktestEngine
from main_backtest import load_intraday, load_daily, calc_daily_bias, apply_v11_filter
from strategy.ensemble import EnsembleEngine

# 探索範囲
BUY_THRESHOLDS = [1.0, 1.2]
SL_ATR_MULTS = [2.0, 2.5, 3.0]

def main():
    print("🚀 最適化開始...")
    with open("config/strategy_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    test_tickers = ["7203.T", "9984.T", "8035.T"]
    
    print("📦 データ読み込み中...")
    raw_signals = {}
    biases = {}
    ensemble = EnsembleEngine("config/strategy_config.yaml")
    
    for t in test_tickers:
        d5 = load_intraday(t)
        dd = load_daily(t)
        if d5 is not None and dd is not None:
            raw_signals[t] = ensemble.generate_ensemble_signals(d5)
            biases[t] = calc_daily_bias(dd, config)

    results = []
    for b_th in BUY_THRESHOLDS:
        for sl_m in SL_ATR_MULTS:
            config["ensemble"]["buy_threshold"] = b_th
            config["exit"]["stop_loss_atr_multiplier"] = sl_m
            
            with open("config/temp_config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(config, f)
            
            current_signals = {}
            for t in raw_signals:
                df = raw_signals[t].copy()
                df["final_signal"] = "HOLD"
                df.loc[df["ensemble_score"] >= b_th, "final_signal"] = "BUY"
                df = apply_v11_filter(df, biases[t])
                current_signals[t] = df
                
            engine = BacktestEngine("config/temp_config.yaml")
            res = engine.run(current_signals)
            
            pnl = sum(t.pnl for t in res.trades)
            print(f"検証: BuyTH={b_th}, SL={sl_m} -> PNL={pnl:+,.0f}")
            results.append({"BuyTH": b_th, "SL": sl_m, "PNL": pnl})

    print("\n🏆 結果一覧")
    print(pd.DataFrame(results).sort_values("PNL", ascending=False))

if __name__ == "__main__":
    main()