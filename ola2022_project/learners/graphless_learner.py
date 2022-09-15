import logging
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

from ola2022_project.environment.environment import (
    MaskedEnvironmentData,
    Interaction,
    alpha_function,
)
from ola2022_project.learners import Learner
from ola2022_project.optimization import budget_assignment, get_expected_value_per_node
from ola2022_project.utils import calculate_aggregated_budget_value


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

    def predict(self, data: MaskedEnvironmentData) -> np.ndarray:
        budget_steps = np.linspace(0, data.total_budget, self.n_budget_steps)

        # Sample current estimation of graph
        graph = self.graph_estimation()

        product_graph_landing_values_for_each_user = np.array(
            [
                get_expected_value_per_node(
                    graph,
                    data.product_prices,
                    [p.reservation_price for p in user_class],  # noqa
                    data.next_products,
                    data.lam,
                )
                for user_class in data.classes_parameters
            ]
        )

        product_graph_landing_values = np.sum(
            product_graph_landing_values_for_each_user, axis=0
        ) / np.sum(product_graph_landing_values_for_each_user)

        logger.debug(product_graph_landing_values)

        # We know the alpha function in the graphless learner, so use it
        # directly here
        class_budget_alphas = np.array(
            [
                [
                    [
                        alpha_function(budget, steepness, shift, upper_bound)
                        for budget in budget_steps
                    ]
                    for (_, steepness, shift, upper_bound) in user_class
                ]
                for user_class in data.classes_parameters
            ]
        )

        aggregated_budget_value_matrix = calculate_aggregated_budget_value(
            product_graph_landing_values=list(product_graph_landing_values),
            product_prices=data.product_prices,
            class_budget_alphas=class_budget_alphas,
            class_ratios=data.class_ratios,
            class_reservation_prices=[
                [float(p.reservation_price) for p in user_class]
                for user_class in data.classes_parameters
            ],
        )

        aggregated_budget_value_matrix = np.array(aggregated_budget_value_matrix)

        best_allocation_index = budget_assignment(aggregated_budget_value_matrix)
        best_allocation = self.budget_steps[best_allocation_index]

        return best_allocation

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
