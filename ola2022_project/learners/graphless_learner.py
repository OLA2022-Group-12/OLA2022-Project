from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

from ola2022_project.environment.environment import MaskedEnvironmentData, Interaction
from ola2022_project.learners import Learner
from ola2022_project.optimization import budget_assignment


class GraphlessLearner(Learner):

    """This class implements an instance of the learner with unknown graph weights."""

    def __init__(self, rng, n_budget_steps, data: MaskedEnvironmentData) -> None:

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

        self.rng = rng
        self.n_budget_steps = n_budget_steps
        self.n_products = len(data.product_prices)
        self.budget_steps = np.linspace(0, data.total_budget, self.n_budget_steps)

        self.alpha_param = np.ones((self.n_products, self.n_products))
        self.beta_param = np.ones((self.n_products, self.n_products))

        # alpha_param = rng.normal(10, 3, (n_products, n_products))
        # beta_param = rng.normal(10, 3, (n_products, n_products))

    def predict(self, data: MaskedEnvironmentData) -> np.ndarray:
        # Sample current estimation of graph
        # graph = self.rng.beta(self.alpha_param, self.beta_param)
        # np.fill_diagonal(graph, 0)

        # TODO remove this when implemented
        return np.full(
            (1, len(data.product_prices)), data.total_budget / len(data.product_prices)
        )

        aggregated_budget_value_matrix = [
            # TODO
        ]

        aggregated_budget_value_matrix = np.array(aggregated_budget_value_matrix)

        best_allocation_index = budget_assignment(aggregated_budget_value_matrix)
        best_allocation = self.budget_steps[best_allocation_index]

        return best_allocation

    def learn(
        self, interactions: List[Interaction], reward: float, prediction: np.ndarray
    ):
        # TODO
        pass

    def show_graph_weight_distributions(self):
        fig, axss = plt.subplots(
            nrows=self.n_products,
            ncols=self.n_products,
            sharex=True,
            sharey=True,
        )
        xs = np.linspace(0, 1, 30)
        for y, axs in enumerate(axss):
            for x, ax in enumerate(axs):
                if x == y:
                    continue
                ax.plot(xs, beta.pdf(xs, self.alpha_param[y, x], self.beta_param[y, x]))

        return fig, axss
