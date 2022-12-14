import logging
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

from ola2022_project.environment.environment import (
    MaskedEnvironmentData,
    Interaction,
    Feature,
)

from ola2022_project.algorithms.reward_estimator import find_optimal_superarm

from ola2022_project.learners import Learner


logger = logging.getLogger(__name__)


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

        self.next_products = data.next_products

    def graph_estimation(self) -> np.ndarray:
        # Sample current estimation of graph
        graph = self.rng.beta(self.alpha_param, self.beta_param)

        # Mask all auto loops
        np.fill_diagonal(graph, 0.0)

        # Mask all non-existent edges
        for product, nexts in enumerate(self.next_products):
            for other_product in range(self.n_products):
                if other_product not in nexts:
                    graph[product, other_product] = 0.0

        return graph

    def predict(
        self, data: MaskedEnvironmentData
    ) -> Tuple[np.ndarray, Optional[List[List[Feature]]]]:
        budget_steps = np.linspace(0, data.total_budget, self.n_budget_steps)

        # Sample current estimation of graph
        graph = self.graph_estimation()

        best_allocation = find_optimal_superarm(data, budget_steps, custom_graph=graph)

        return budget_steps[best_allocation], None

    def learn(
        self, interactions: List[Interaction], reward: float, prediction: np.ndarray
    ):
        for interaction in interactions:
            remaining_edges = [
                (product, next_)
                for product, nexts in enumerate(self.next_products)
                for next_ in nexts
                # Only add edge to remaining if we visited the page, otherwise
                # we got no information about those edges
                if interaction.items_bought[product] > 0
            ]
            for edge in interaction.edges:
                s, t = edge
                self.alpha_param[s, t] += 1
                remaining_edges.remove(edge)

            for edge in remaining_edges:
                s, t = edge
                self.beta_param[s, t] += 1

    def show_progress(self, fig: plt.Figure):
        axss = fig.subplots(
            nrows=self.n_products,
            ncols=self.n_products,
            sharex=True,
            sharey=True,
        )
        actual_edges = [
            (product, next_)
            for product, nexts in enumerate(self.next_products)
            for next_ in nexts
        ]
        xs = np.linspace(0, 1, 30)
        for y, axs in enumerate(axss):
            for x, ax in enumerate(axs):
                ax: plt.Axes = ax
                if (x, y) not in actual_edges:
                    # if x == y:
                    ax.axis("off")
                    continue

                alpha_v = self.alpha_param[y, x]
                beta_v = self.beta_param[y, x]
                # ax.set_ylim(0.0, 1.0)
                ax.set_xlim(0.0, 1.0)
                ax.plot(xs, beta.pdf(xs, alpha_v, beta_v))
                ax.set_title(f"{x} -> {y}")

        fig.set_tight_layout(True)
        fig.suptitle(
            "Probability distributions of adjacency matrix\n(edge between product x -> y)"
        )
