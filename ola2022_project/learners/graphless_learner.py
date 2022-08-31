import numpy as np
from ola2022_project.environment.environment import MaskedEnvironmentData
from ola2022_project.learners import Learner
from ola2022_project.optimization import budget_assignment


class GraphlessLearner(Learner):

    """This class implements an instance of the learner with unknown graph weights."""

    def __init__(self, n_budget_steps, data: MaskedEnvironmentData) -> None:

        """Creates a learner which works on unknown graph weights

        Arguments:
            n_budget_steps: total number of individual budget steps

            data: instance of a MaskedEnvironmentData initialized with Step.ONE

        Returns:
            An GraphlessLearner instance, which adopts the same framework as the
            parent class Learner

        """

        if (
            data.classes_parameters is None
            or data.graph is not None
            or data.class_ratios is None
        ):
            raise RuntimeError(
                "Graph-less learner called with wrong masked environment"
            )

        self.n_budget_steps = n_budget_steps
        self.n_products = len(data.product_prices)
        self.budget_steps = np.linspace(0, data.total_budget, self.n_budget_steps)

    def predict(self, data: MaskedEnvironmentData) -> np.ndarray:
        aggregated_budget_value_matrix = [
            # TODO
        ]

        aggregated_budget_value_matrix = np.array(aggregated_budget_value_matrix)

        best_allocation_index = budget_assignment(aggregated_budget_value_matrix)
        best_allocation = self.budget_steps[best_allocation_index]

        return best_allocation

    def learn(self, reward: float, prediction: np.ndarray):
        # TODO
        pass
