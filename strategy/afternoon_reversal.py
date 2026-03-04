"""
午後リバーサル��略（アフタヌーン・リバーサル）
- 午前中に大きく動いた銘柄の「戻し」を逆張りで狙う
- RSI + ボリンジャーバンド + VWAP回帰
- 空売り対応
"""

import pandas as pd
import pandas_ta as ta
import numpy as np


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """当日VWAPを計算"""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum()
    vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
    return vwap


def calc_morning_move(df_day: pd.DataFrame) -> dict:
    """
    各日の午前変動率を計算
    戻り値: {date: {"open": 始値, "morning_high": 午前高値, "morning_low": 午前安値, "move_pct": 変動率%}}
    """
    result = {}
    for date_val in df_day.index.map(lambda x: x.date()).unique():
        day_data = df_day[df_day.index.map(lambda x: x.date()) == date_val]

        # 午前データ（9:00〜11:30）
        morning = day_data[day_data.index.map(lambda x: x.hour * 100 + x.minute) <= 1130]
        if len(morning) < 2:
            continue

        open_price = float(morning["open"].iloc[0])
        morning_high = float(morning["high"].max())
        morning_low = float(morning["low"].min())

        if open_price > 0:
            move_pct = ((morning_high - morning_low) / open_price) * 100
        else:
            move_pct = 0

        result[date_val] = {
            "open": open_price,
            "morning_high": morning_high,
            "morning_low": morning_low,
            "move_pct": move_pct,
        }
    return result


def generate_afternoon_signals(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    午後リバーサルシグナルを生成

    Parameters:
        df: 5分足OHLCV（1銘柄分、インデックスがDatetime）
        config: afternoon_config.yaml の内容

    Returns:
        df に "afternoon_signal" 列を追加したDataFrame
    """
    p = config.get("afternoon_reversal", {})
    rsi_period = p.get("rsi_period", 14)
    rsi_oversold = p.get("rsi_oversold", 25)
    rsi_overbought = p.get("rsi_overbought", 75)
    bb_period = p.get("bb_period", 20)
    bb_std = p.get("bb_std", 2.0)
    morning_move_threshold = p.get("morning_move_pct", 1.5)
    volume_ratio_threshold = p.get("volume_ratio", 1.3)
    vwap_distance_pct = p.get("vwap_distance_pct", 0.5)

    df = df.copy()

    # インジケーター計算
    df["rsi"] = ta.rsi(df["close"], length=rsi_period)

    bb = ta.bbands(df["close"], length=bb_period, std=bb_std)
    if bb is not None:
        bb.columns = ["bb_lower", "bb_mid", "bb_upper", "bb_bw", "bb_pct"]
        df = pd.concat([df, bb], axis=1)
    else:
        df["bb_lower"] = np.nan
        df["bb_mid"] = np.nan
        df["bb_upper"] = np.nan

    # 出来高移動平均
    df["vol_ma"] = df["volume"].rolling(window=20).mean()

    # VWAP（日ごとにリセット）
    df["vwap"] = np.nan
    for date_val in df.index.map(lambda x: x.date()).unique():
        mask = df.index.map(lambda x: x.date()) == date_val
        day_df = df.loc[mask]
        if len(day_df) > 0:
            df.loc[mask, "vwap"] = calc_vwap(day_df).values

    # 午前変動率の計算
    morning_moves = calc_morning_move(df)

    # シグナル生成
    df["afternoon_signal"] = "HOLD"

    for idx in df.index:
        date_val = idx.date()
        hour_min = idx.hour * 100 + idx.minute

        # 午後セッションのみ（12:30〜14:00）
        if hour_min < 1230 or hour_min > 1400:
            continue

        # 午前変動率チェック
        mm = morning_moves.get(date_val)
        if mm is None or mm["move_pct"] < morning_move_threshold:
            continue

        row_idx = df.index.get_loc(idx)
        close = df["close"].iloc[row_idx]
        rsi_val = df["rsi"].iloc[row_idx]
        bb_lower = df["bb_lower"].iloc[row_idx]
        bb_upper = df["bb_upper"].iloc[row_idx]
        vol = df["volume"].iloc[row_idx]
        vol_ma = df["vol_ma"].iloc[row_idx]
        vwap_val = df["vwap"].iloc[row_idx]

        # NaNチェック
        if any(pd.isna(v) for v in [rsi_val, bb_lower, bb_upper, vol_ma, vwap_val]):
            continue

        # 出来高フィルター
        if vol_ma > 0 and vol < vol_ma * volume_ratio_threshold:
            continue

        # VWAP乖離率
        vwap_dist = abs(close - vwap_val) / vwap_val * 100
        if vwap_dist < vwap_distance_pct:
            continue

        # BUY: 売られすぎ → 反発狙い
        if rsi_val <= rsi_oversold and close < bb_lower and close < vwap_val:
            df.at[idx, "afternoon_signal"] = "BUY"

        # SELL: 買われすぎ → 反落狙い（空売り）
        elif rsi_val >= rsi_overbought and close > bb_upper and close > vwap_val:
            df.at[idx, "afternoon_signal"] = "SELL"

    return df
