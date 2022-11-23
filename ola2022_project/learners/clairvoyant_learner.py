import logging
from typing import List, Tuple, Optional
from ola2022_project.utils import calculate_aggregated_budget_value
from ola2022_project.learners import Learner
from ola2022_project.environment.environment import (
    MaskedEnvironmentData,
    alpha_function,
    Feature,
)
from ola2022_project.optimization import budget_assignment
import numpy as np
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class ClairvoyantLearner(Learner):
    """Learner that always predicts the optimal delegation of budgets

    NB! It has to be called with ALL data avaliable in MaskedEnvironmentData
    """

    def __init__(self, n_budget_steps=5) -> None:
        self.n_budget_steps = n_budget_steps

    def learn(self, interactions, reward, prediction):
        pass

    def predict(
        self, data: MaskedEnvironmentData
    ) -> Tuple[np.ndarray, Optional[List[List[Feature]]]]:
        if (
            data.classes_parameters is None
            or data.class_ratios is None
            or data.graph is None
        ):
            raise RuntimeError("ClairvoyantLearner called with masked env parameters")

        n_products = len(data.product_prices)

        # NB! Budget steps always have to be evenly spaced for the budget
        # assignment algorithm to work correctly (I think...)
        budget_steps = np.linspace(0, data.total_budget, self.n_budget_steps)

        # TODO make call to graph influence algorithm which should return the
        # relative value of landing on each node/product in the graph. I.e. if
        # landing on product 1 more frequently leads to buying product 2 and 3
        # aswell, it should have a high value here.
        product_graph_landing_values = np.ones((n_products,))

        # In the clairvoyant learner we know these alpha values exactly,
        # hence here it is trivial to calculate the best allocation based on
        # this. In the other learners this is the function we need to learn.
        class_budget_alphas = np.array(
            [
                [
                    [
                        alpha_function(budget, upper_bound, max_useful_budget)
                        for budget in budget_steps
                    ]
                    for (_, upper_bound, max_useful_budget) in user_class
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

        best_allocation_index = budget_assignment(aggregated_budget_value_matrix)

        # The budget_assignment algorithm gives us back the index into the
        # budget matrix, hence we have to convert it back to the actual budgets
        # that we should assign before returning
        best_allocation = budget_steps[best_allocation_index]  # type: ignore

        logger.debug(f"best_allocation {best_allocation.shape}: {best_allocation}")
        logger.debug(
            f"sum_of_allocations / total_budget = {best_allocation.sum()} / {data.total_budget}"
        )

        return best_allocation, None

    def show_progress(self, fig: plt.Figure):
        pass
