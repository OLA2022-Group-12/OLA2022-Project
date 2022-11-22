from typing import List, Tuple
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from ola2022_project.environment.environment import (
    MaskedEnvironmentData,
    Interaction,
    Feature,
)


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
    def predict(
        self, data: MaskedEnvironmentData
    ) -> Tuple[np.ndarray, List[List[Feature]]]:

        """Makes an inference about the values of the budgets for the subcampaigns
        utilizing the information gathered over time and the current state of the
        environment

        Arguments:
            data: up-to-date, complete or incomplete environment information that is
                used by the learner in order to make the inference

        Returns:
            a tuple containing a list of values (corresponding to the budgets inferred given
            the knowledge obtained by the learner until now) and a list of features (referring
            to which particular customers were the budgets aimed for, if None, the budgets
            apply to all the customers)
        """

        pass

    @abstractmethod
    def show_progress(self, fig: plt.Figure):

        """Creates a figure and plots showing the status of learning progress

        Arguments:
            fig: a matplotlib figure which will be filled with subplots/plots
        """
        pass
