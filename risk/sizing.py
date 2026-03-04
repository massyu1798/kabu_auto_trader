"""ポジションサイジング計算"""


def calc_position_size(
    capital: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_price: float,
) -> int:
    risk_amount = capital * risk_pct
    sl_distance = abs(entry_price - stop_loss_price)

    if sl_distance <= 0:
        return 0

    size = int(risk_amount / sl_distance)
    return max(size, 0)


def calc_max_position_value(
    capital: float,
    max_positions: int,
) -> float:
    return capital / max_positions