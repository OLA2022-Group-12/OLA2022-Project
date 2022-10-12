from ola2022_project.learners.learner import Learner
from ola2022_project.environment.environment import MaskedEnvironmentData
import numpy as np
import matplotlib.pyplot as plt


class StupidLearner(Learner):
    """Learner that always subdivides equally the budget between products and
    doesn't update its prediction.
    """

    def __init__(self) -> None:
        pass

    def learn(self, interactions, reward, prediction):
        pass

    def predict(self, data: MaskedEnvironmentData) -> np.ndarray:
        return np.full(
            (len(data.product_prices),), data.total_budget / len(data.product_prices)
        )

    def show_progress(self, fig: plt.Figure):
        pass
