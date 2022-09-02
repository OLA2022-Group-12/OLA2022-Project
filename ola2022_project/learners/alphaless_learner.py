from typing import List
import numpy as np
from ola2022_project.environment.environment import (
    AggregatedInteraction,
    MaskedEnvironmentData,
    Step,
)
from ola2022_project.learners import Learner
from ola2022_project.algorithms.multi_armed_bandits import (
    GPTSLearner,
    GPUCB1Learner,
    Mab,
)
from ola2022_project.optimization import budget_assignment


def create_alphaless_learner(
    rng, step: Step, n_budget_steps, data: MaskedEnvironmentData, mab_algorithm=Mab.GPTS
):
    if step == Step.ONE:
        return AlphaUnitslessLearner(rng, n_budget_steps, data, mab_algorithm)

    elif step == Step.TWO:
        return AlphalessLearner(rng, n_budget_steps, data, mab_algorithm)

    else:
        raise RuntimeError(
            "Cannot create alphaless learner with the Step provided. Step must be ONE or TWO"
        )


class AlphaUnitslessLearner(Learner):

    """This class implements an instance of the learner with unknown alpha functions
    and number of units sold per product. The learner will receive aggregated data
    so it must work by estimating aggregated alpha functions.
    """

    def __init__(
        self, rng, n_budget_steps, data: MaskedEnvironmentData, mab_algorithm=Mab.GPTS
    ) -> None:

        """Creates a learner which works on unknown alpha functions.

        Arguments:
            rng: numpy generator (such as default_rng)

            n_budget_steps: total number of individual budget steps

            data: instance of a MaskedEnvironmentData initialized with Step.ONE

            mab_algorithm: specifices whether the learner should be implemented
                with a Gaussian Process Thompson Sampling or Gaussian Process
                UCB1

        Returns:
            An AlphalessLearner instance, which adopts the same framework as the
            parent class Learner

        """

        if (
            data.classes_parameters is not None
            or data.graph is None
            or data.class_ratios is None
        ):
            raise RuntimeError(
                "Alpha-less learner called with wrong masked environment"
            )

        self.n_budget_steps = n_budget_steps
        self.n_products = len(data.product_prices)
        self.budget_steps = np.linspace(0, data.total_budget, self.n_budget_steps)
        self.total_budget = data.total_budget
        self.env = data

        normalized_budget_steps = self.budget_steps / data.total_budget

        # According to the specified algorithm the functions creates 5 GP
        # learners, one for each product

        if mab_algorithm == Mab.GPTS:
            self.product_mabs = [
                GPTSLearner(rng, self.n_budget_steps, normalized_budget_steps)
                for _ in range(self.n_products)
            ]

        elif mab_algorithm == Mab.GPUCB1:
            self.product_mabs = [
                GPUCB1Learner(rng, self.n_budget_steps, normalized_budget_steps)
                for _ in range(self.n_products)
            ]

    def predict(self, data: MaskedEnvironmentData) -> np.ndarray:
        aggregated_budget_value_matrix = [
            self.product_mabs[i].estimation() for i in range(len(self.product_mabs))
        ]

        aggregated_budget_value_matrix = np.array(aggregated_budget_value_matrix)

        best_allocation_index = budget_assignment(aggregated_budget_value_matrix)
        best_allocation = self.budget_steps[best_allocation_index]

        return best_allocation

    def learn(
        self, interactions: List[Interaction], reward: float, prediction: np.ndarray
    ):
        for i, p in enumerate(prediction):
            prediction_index = np.where(self.budget_steps == p)[0][0]
            self.product_mabs[i].update(prediction_index, reward)


class AlphalessLearner(AlphaUnitslessLearner):
    def __init__(
        self, rng, n_budget_steps, data: MaskedEnvironmentData, mab_algorithm=Mab.GPTS
    ) -> None:
        super().__init__(rng, n_budget_steps, data, mab_algorithm)

    def learn(self, interactions: List[AggregatedInteraction], prediction: np.ndarray):
        if not isinstance(interactions, list):
            raise RuntimeError(
                """Alpha-less learner cannot learn from aggregate reward,
                it needs a list of interactions."""
            )

        rewards = self._compute_products_rewards(interactions, self.env.product_prices)

        for mab, pred, rew in zip(self.product_mabs, prediction, rewards):
            prediction_index = np.where(self.budget_steps == pred)[0][0]
            mab.update(prediction_index, rew)

    def _compute_products_rewards(
        self, interactions: List[AggregatedInteraction], product_prices
    ):

        reward_per_product = list()

        for product in range(len(product_prices)):

            customers_landed_on_product = list(
                filter(lambda x: x.landed_on == product, interactions)
            )

            units = [e.items_bought for e in customers_landed_on_product]

            if not units:
                units = np.zeros((1, self.n_products))

            reward_per_product.append(np.dot(np.sum(units, axis=0), product_prices))

        return reward_per_product
