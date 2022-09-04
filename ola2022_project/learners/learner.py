from typing import List
from abc import ABC, abstractmethod

import numpy as np

from ola2022_project.environment.environment import MaskedEnvironmentData, Interaction


class Learner(ABC):

    """Generic Learner interface for interactive agents capable of learning the budget
    distribution for a set of subcampaigns from the environment they exist in and the
    online feedback returned.
    """

    @abstractmethod
    def learn(
        self, interactions: List[Interaction], reward: float, prediction: np.ndarray
    ):

        """Updates the learner's properties according to the reward received.

        Arguments:
            interactions: the interactions of the users which led to the given
            reward

            reward: the reward obtained from the environment based on the
            prediction given, needed for the tuning of internal properties done
            by the learner

            prediction: array containing the previous budget evaluation of the learner
        """

        pass

    @abstractmethod
    def predict(self, data: MaskedEnvironmentData) -> np.ndarray:

        """Makes an inference about the values of the budgets for the subcampaigns
        utilizing the information gathered over time and the current state of the
        environment

        Arguments:
            data: up-to-date, complete or incomplete environment information that is
                used by the learner in order to make the inference

        Returns:
            a list of values, corresponding to the budgets inferred given the knowledge
            obtained by the learner until now
        """

        pass
