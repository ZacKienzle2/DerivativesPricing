# models/pricers/base_pricer.py

from abc import ABC, abstractmethod
from typing import Tuple
from models.options import BaseOption


class BasePricer(ABC):
    """
    An abstract base class for option pricing models.
    """

    def __init__(self, option: BaseOption):
        """Initialises the pricer with a specific option contract."""
        self.option = option

    @abstractmethod
    def price(self) -> Tuple[float, float]:
        """
        Calculates the theoretical price of the option.

        Returns:
            A tuple containing the price and the standard error. For analytical
            models, the standard error will be 0.0.
        """
        pass
