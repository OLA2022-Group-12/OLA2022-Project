from typing import List, Tuple, Optional
import numpy as np
from ola2022_project.environment.environment import (
    AggregatedInteraction,
    MaskedEnvironmentData,
    Feature,
)
from ola2022_project.learners import Learner
from ola2022_project.algorithms.multi_armed_bandits import (
    GPTSLearner,
    GPUCB1Learner,
    Mab,
)
from ola2022_project.optimization import budget_assignment
import matplotlib.pyplot as plt


class AlphaUnitslessLearner(Learner):

    """This class implements an instance of the learner with unknown alpha functions
    and number of units sold per product. The learner will receive aggregated data
    so it must work by estimating aggregated alpha functions.
    """

    def __init__(
        self,
        rng,
        n_budget_steps,
        data: MaskedEnvironmentData,
        mab_algorithm=Mab.GPTS,
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
        self.rng = rng

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
                GPUCB1Learner(
                    rng, self.n_budget_steps, normalized_budget_steps, confidence=1
                )
                for _ in range(self.n_products)
            ]

    def predict(
        self, data: MaskedEnvironmentData
    ) -> Tuple[np.ndarray, Optional[List[List[Feature]]]]:
        aggregated_budget_value_matrix = [
            self.product_mabs[i].estimation() for i in range(len(self.product_mabs))
        ]

        aggregated_budget_value_matrix = np.array(aggregated_budget_value_matrix)
        best_allocation_index = budget_assignment(aggregated_budget_value_matrix)
        best_allocation = self.budget_steps[best_allocation_index]

        return best_allocation, None

    def learn(self, _, reward: float, prediction: np.ndarray):
        for i, p in enumerate(prediction):
            prediction_index = np.where(self.budget_steps == p)[0][0]
            self.product_mabs[i].update(prediction_index, reward)

    def slide_window(self):
        for mab in self.product_mabs:
            mab.delete_first_observation()

    def reset(self):
        algorithm = self.product_mabs[0].__class__

        normalized_budget_steps = self.budget_steps / self.env.total_budget

        self.product_mabs = [
            algorithm(self.rng, self.n_budget_steps, normalized_budget_steps)
            for _ in range(self.n_products)
        ]

    def show_progress(self, fig: plt.Figure):
        pass


class AlphalessLearner(AlphaUnitslessLearner):

    """This class implements an instance of the learner with unknown alpha functions.
    The learner will receive all the interactions minus the class data. I will then
    try to estimate the reward given by every single product.
    """

    def __init__(
        self, rng, n_budget_steps, data: MaskedEnvironmentData, mab_algorithm=Mab.GPTS
    ) -> None:
        super().__init__(rng, n_budget_steps, data, mab_algorithm)

    def predict_raw(self, data: MaskedEnvironmentData) -> np.ndarray:

        """Acts as the predict function but doesn't pass the result through the optimizer
        and returns the aggregated budget value matrix.

        Arguments:
            data: up-to-date, complete or incomplete environment information that is
                used by the learner in order to make the inference

        Returns:
            a list of values, corresponding to an estimation of the earnings for each
            product given the knowledge obtained by the learner until now
        """

        aggregated_budget_value_matrix = [
            self.product_mabs[i].estimation() for i in range(len(self.product_mabs))
        ]

        return np.array(aggregated_budget_value_matrix)

    def learn(
        self, interactions: List[AggregatedInteraction], _, prediction: np.ndarray
    ):
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
    ) -> list:

        """This function, given the interactions, compute the reward that every
        single product generated.

        Arguments:
            interactions: list contaning all the daily interactions. It should be a
                list of aggregated interactions.

            product_prices: array containing prices of every product.

        Return:
            A list where every element is the reward generated by product i+1
        """

        reward_per_product = list()

        # We separately consider all the interactions where users landed on
        # a certain product, then compute the reward they generated.
        # This way we obtain reward associated to a single campaign.
        for product in range(len(product_prices)):

            customers_landed_on_product = list(
                filter(lambda x: x.landed_on == product, interactions)
            )

            units = [e.items_bought for e in customers_landed_on_product]

            # Simple check to avoid that an empty array messes with the dot product
            if not units:
                units = np.zeros((1, self.n_products))

            reward_per_product.append(np.dot(np.sum(units, axis=0), product_prices))

        return reward_per_product
