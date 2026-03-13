"""ポジションサイジング計算 v2.0: ATRベース + レバレッジ対応

ATRベースのポジションサイジング（推奨）:
  ポジションサイズ（株数）= 最大損失額 / (ATR(14) × 倍率)

例:
  - 最大損失: 100,000円
  - ATR(14): 50円
  - 倍率: 2.0（ATRの2倍を損切り幅に設定）
  - → 100,000 / (50 × 2) = 1,000株
"""

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


def calc_position_size_atr(
    capital: float,
    risk_pct: float,
    atr: float,
    atr_multiplier: float = 2.0,
    lot_unit: int = 100,
) -> int:
    """
    ATRベースのポジションサイジング（推奨）。

    ポジションサイズ = (capital × risk_pct) / (ATR × atr_multiplier)

    引数:
      capital       : 有効資本（円）
      risk_pct      : 1トレード最大損失率（例: 0.01 = 1%）
      atr           : ATR(14)の値（円）
      atr_multiplier: ATRの何倍を損切り幅とするか（default: 2.0）
      lot_unit      : 単元株数（default: 100）

    戻り値: 株数（lot_unit の倍数に丸め）
    """
    if atr <= 0 or capital <= 0:
        return 0
    max_loss = capital * risk_pct
    sl_distance = atr * atr_multiplier
    size = int(max_loss / sl_distance)
    size = (size // lot_unit) * lot_unit
    return max(size, 0)


def apply_position_caps(
    size: int,
    entry_price: float,
    capital: float,
    max_position_pct: float = _MAX_EXPOSURE_PER_POSITION_RATIO,
    lot_unit: int = 100,
) -> int:
    """
    ポジションサイズに1銘柄最大建玉キャップを適用する。

    引数:
      size             : 元のポジションサイズ（株数）
      entry_price      : エントリー価格
      capital          : 有効資本（信用枠含む）
      max_position_pct : 1銘柄最大建玉率（default: 15%）
      lot_unit         : 単元株数

    戻り値: キャップ適用後のポジションサイズ
    """
    if entry_price <= 0:
        return 0
    max_by_cap = int(capital * max_position_pct / entry_price // lot_unit) * lot_unit
    return max(min(size, max_by_cap), 0)