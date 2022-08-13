import numpy as np
from ola2022_project.environment.environment import MaskedEnvironmentData
from ola2022_project.learners import Learner
from ola2022_project.learners.MAB_algorithms import GPTSLearner, Mab
from ola2022_project.optimization import budget_assignment


class AlphalessLearner(Learner):
    def __init__(
        self, rng, n_budget_steps, data: MaskedEnvironmentData, mab_algorithm=Mab.GPTS
    ) -> None:
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

        normalized_budget_steps = self.budget_steps / data.total_budget

        if mab_algorithm == Mab.GPTS:
            self.product_mabs = [
                GPTSLearner(rng, self.n_budget_steps, normalized_budget_steps)
                for _ in range(self.n_products)
            ]

        elif mab_algorithm == Mab.GPUCB1:
            # TODO: Implement GPUCB1
            pass

    def predict(self, data: MaskedEnvironmentData) -> np.ndarray:
        aggregated_budget_value_matrix = [
            self.product_mabs[i].estimation() for i in range(len(self.product_mabs))
        ]

        aggregated_budget_value_matrix = np.array(aggregated_budget_value_matrix)

        best_allocation_index = budget_assignment(aggregated_budget_value_matrix)
        best_allocation = self.budget_steps[best_allocation_index]

        return best_allocation

    def learn(self, reward: float, prediction: np.ndarray):
        for i, p in enumerate(prediction):
            prediction_index = np.where(self.budget_steps == p)[0][0]
            self.product_mabs[i].update(prediction_index, reward)
