import numpy as np
import matplotlib.pyplot as plt
from typing import List
from ola2022_project.learners import Learner
from ola2022_project.algorithms.context_splitting import (
    Context,
    partial_tree_generation,
)
from ola2022_project.simulation.simulation import Simulation
from ola2022_project.environment.environment import (
    AggregatedInteraction,
    MaskedEnvironmentData,
    Feature,
    Step,
)
from ola2022_project.algorithms.multi_armed_bandits import Mab
from ola2022_project.optimization import budget_assignment


class ContextualLearner(Learner):

    """This class implements an instance of the learner with unknown alpha functions
    and number of units sold per product. The learner will receive aggregated data
    so it must work by estimating aggregated alpha functions.
    """

    def __init__(
        self,
        rng,
        n_budget_steps,
        data: MaskedEnvironmentData,
        simulation: Simulation,
        features: List[Feature],
        mab_algorithm=Mab.GPTS,
    ) -> None:

        """Creates a learner which works on unknown alpha functions.

        Arguments:
            rng: numpy generator (such as default_rng)

            n_budget_steps: total number of individual budget steps

            data: instance of a MaskedEnvironmentData initialized with Step.ONE

            simulation: TODO

            features: TODO

            mab_algorithm: specifices whether the learner should be implemented
                with a Gaussian Process Thompson Sampling or Gaussian Process
                UCB1

        Returns:
            An AlphalessLearner instance, which adopts the same framework as the
            parent class Learner

        """

        self.rng = rng
        self.n_budget_steps = n_budget_steps
        self.n_products = len(data.product_prices)
        self.budget_steps = np.linspace(0, data.total_budget, self.n_budget_steps)
        self.total_budget = data.total_budget
        self.env = data
        self.mab_algorithm = mab_algorithm

        self.features = features
        self.simulation = simulation.copy(include_learner=False)
        self.simulation.step = Step.ONE
        self.learners = [self.simulation.learner]
        self.contexts = [Context(self.simulation, [], 0, 0, 0, 0)]

    def predict(self, _) -> np.ndarray:
        aggregated_budget_value_matrix = [
            np.array(learner.predict_raw(self.env)) for learner in self.learners
        ]
        aggregated_budget_value_matrix = np.vstack(aggregated_budget_value_matrix)
        best_allocation_index = budget_assignment(aggregated_budget_value_matrix)
        best_allocation = self.budget_steps[best_allocation_index]

        return best_allocation

    def learn(
        self, interactions: List[AggregatedInteraction], _, prediction: np.ndarray
    ):
        for i, learner in enumerate(self.learners):
            lower = i * self.n_products
            upper = (i + 1) * self.n_products
            learner.learn(
                interactions,
                _,
                prediction[lower:upper],
            )

    def context_generation(self, dataset):

        """TODO"""

        self.contexts = partial_tree_generation(
            self.simulation,
            dataset,
            self.features,
            self.contexts,
        )
        self.learners = [context.learner_sim.learner for context in self.contexts]

    def show_progress(self, fig: plt.Figure):
        pass
