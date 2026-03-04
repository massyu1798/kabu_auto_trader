from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

@dataclass
class Signal:
    score: float
    reason: str

class StrategyBase(ABC):
    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        pass