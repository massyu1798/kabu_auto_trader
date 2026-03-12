"""ポジションサイジング計算 v14: レバレッジ対応"""

# 1ポジションあたりの最大建玉上限（有効資本に対する割合）
_MAX_EXPOSURE_PER_POSITION_RATIO = 0.15


def calc_position_size(
    capital: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_price: float,
    lot_unit: int = 100,
) -> int:
    """ATRベースのポジションサイジング（100株単位に丸め）"""
    risk_amount = capital * risk_pct
    sl_distance = abs(entry_price - stop_loss_price)

    if sl_distance <= 0:
        return 0

    size = int(risk_amount / sl_distance)
    size = (size // lot_unit) * lot_unit
    return max(size, 0)


def calc_position_size_leveraged(
    margin_capital: float,
    leverage: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_price: float,
    lot_unit: int = 100,
) -> int:
    """レバレッジ考慮のポジションサイジング"""
    effective_capital = margin_capital * leverage
    risk_amount = margin_capital * risk_pct
    sl_distance = abs(entry_price - stop_loss_price)

    if sl_distance <= 0:
        return 0

    size = int(risk_amount / sl_distance)
    size = (size // lot_unit) * lot_unit

    # 1ポジションが建玉上限を超えないよう制限
    max_size_by_exposure = int(effective_capital * _MAX_EXPOSURE_PER_POSITION_RATIO / entry_price)
    max_size_by_exposure = (max_size_by_exposure // lot_unit) * lot_unit

    return max(min(size, max_size_by_exposure), 0)


def calc_max_position_value(
    capital: float,
    max_positions: int,
) -> float:
    return capital / max_positions