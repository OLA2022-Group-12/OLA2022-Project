from ola2022_project.learners.learner import Learner
from ola2022_project.environment.environment import MaskedEnvironmentData
import numpy as np


class StupidLearner(Learner):
    """Learner that always subdivides equally the budget between products and
    doesn't update its prediction.
    """

    def learn(self, _reward, _prediction):
        pass

    def predict(self, data: MaskedEnvironmentData) -> np.ndarray:
        return np.full(
            (1, len(data.product_prices)), data.total_budget / len(data.product_prices)
        )