from typing import List
import numpy as np
from ola2022_project.algorithms.reward_estimator import compute_user_influence
from ola2022_project.environment.environment import (
    Interaction,
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

    """This class implements an instance of the learner with unknown alpha functions.
    It works with aggregated data (regarding classes) but can observe all the daily
    interactions, so the reward must not be aggregated.
    """

    def __init__(
        self,
        rng,
        n_budget_steps,
        data: MaskedEnvironmentData,
        mab_algorithm=Mab.GPTS,
        estimation_accuracy=10,
    ) -> None:

        super().__init__(rng, n_budget_steps, data, mab_algorithm)
        self.unaffordable_ratio = np.ones(self.n_products)
        self.estimation_accuracy = estimation_accuracy

    def predict(self, data: MaskedEnvironmentData) -> np.ndarray:

        landing_values = [
            self.product_mabs[i].estimation() for i in range(len(self.product_mabs))
        ]

        aggregated_budget_value_matrix = (
            np.array(landing_values) * self.product_reward_multiplier.T
        )

        best_allocation_index = budget_assignment(aggregated_budget_value_matrix)
        best_allocation = self.budget_steps[best_allocation_index]

        return best_allocation

    def learn(self, interactions: List[Interaction], prediction: np.ndarray):

        if not isinstance(interactions, list):
            raise RuntimeError(
                """Alpha-less learner cannot learn from aggregate reward,
                it needs a list of interactions."""
            )

        for i, p in enumerate(prediction):
            prediction_index = np.where(self.budget_steps == p)[0][0]
            self.product_mabs[i].update(prediction_index, interactions)

        if self.t == 1:
            self.unaffordable_ratio = self._compute_unaffordable_ratio(interactions)
            self.user_influence = compute_user_influence(
                self.unaffordable_ratio, self.env
            )
            self.product_reward_multiplier = np.atleast_2d(
                compute_user_influence(
                    self.n_products, self.unaffordable_ratio, self.env
                )
            )

        elif self.t <= self.estimation_accuracy:
            self.unaffordable_ratio += (
                self._compute_unaffordable_ratio(interactions) - self.unaffordable_ratio
            ) / self.t
            self.product_reward_multiplier = np.atleast_2d(
                compute_user_influence(
                    self.n_products, self.unaffordable_ratio, self.env
                )
            )

    def _compute_unaffordable_ratio(self, interactions: List[Interaction]):
        unaffordable_ratio = np.zeros(self.n_products)

        for prod in range(self.n_products):
            customers_landed_on_product = list(
                filter(lambda x: x.landed_on == prod, interactions)
            )
            customers_not_buying_product = list(
                filter(lambda x: x.items_bought[prod] == 0, customers_landed_on_product)
            )

            unaffordable_ratio[prod] = len(customers_not_buying_product) / len(
                customers_landed_on_product
            )

        return unaffordable_ratio
