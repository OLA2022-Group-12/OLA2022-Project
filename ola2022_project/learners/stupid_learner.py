from ola2022_project.learners.learner import Learner
from ola2022_project.environment.environment import MaskedEnvironmentData
import numpy as np


class StupidLearner(Learner):
    """Learner that always subdivides equally the budget between products and
    doesn't update its prediction.
    """

    def __init__(self, _=True) -> None:
        pass

    def learn(self, interactions, reward, prediction):
        pass

    def predict(self, data: MaskedEnvironmentData) -> np.ndarray:
        return np.full(
            (1, len(data.product_prices)), data.total_budget / len(data.product_prices)
        )
