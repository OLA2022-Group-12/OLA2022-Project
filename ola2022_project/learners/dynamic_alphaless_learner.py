import logging
from ola2022_project.learners import AlphaUnitslessLearner, AlphalessLearner, Learner
from ola2022_project.environment.environment import (
    Interaction,
    MaskedEnvironmentData,
    Feature,
)
from ola2022_project.algorithms.multi_armed_bandits import Mab
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class SlidingWindowLearner(Learner):

    """This class is a wrapper of an AlphalessLearner or an AlphaUnitlessLearner
    that implements a sliding window behaviour.
    """

    def __init__(
        self,
        rng,
        n_budget_steps,
        data: MaskedEnvironmentData,
        mab_algorithm=Mab.GPTS,
        window_size=10,
        units_less=False,
    ) -> None:

        """Creates a sliding window learner which works on unknown alpha functions.

        Arguments:
            rng: numpy generator (such as default_rng)

            n_budget_steps: total number of individual budget steps

            data: instance of a MaskedEnvironmentData initialized with Step.ONE

            mab_algorithm: specifices whether the learner should be implemented
                with a Gaussian Process Thompson Sampling or Gaussian Process
                UCB1

            window_size: integer representing the mnumber of maximum elements
                inside the slisind window

            units_less: if set to True the implemented learner will be an
                AlphaUnitslessLearner instance, otherwise it's a
                AlphalessLearner instance

        Returns:
            A SlidingWindowLearner instance, which adopts the same framework as the
            parent class Learner

        """

        if units_less:
            self.learner = AlphaUnitslessLearner(
                rng, n_budget_steps, data, mab_algorithm
            )
        else:
            self.learner = AlphalessLearner(rng, n_budget_steps, data, mab_algorithm)

        self.window_size = window_size
        self.t = 0

    def learn(
        self, interactions: List[Interaction], reward: float, prediction: np.ndarray
    ):

        self.t += 1

        # Start sliding window only when t is greater than window size
        if self.t > self.window_size:
            self.learner.slide_window()

        self.learner.learn(interactions, reward, prediction)

    def predict(
        self, data: MaskedEnvironmentData
    ) -> Tuple[np.ndarray, Optional[List[List[Feature]]]]:
        return self.learner.predict(data)

    def show_progress(self, _):
        pass


class ChangeDetectionLearner(Learner):

    """This class is a wrapper of an AlphalessLearner or an AlphaUnitlessLearner
    that implements a change detection behaviour.
    """

    def __init__(
        self,
        rng,
        n_budget_steps,
        data: MaskedEnvironmentData,
        mab_algorithm=Mab.GPTS,
        threshold=500,
        threshold_window=4,
        units_less=False,
    ) -> None:

        """Creates a change detection learner which works on unknown alpha functions.

        Arguments:
            rng: numpy generator (such as default_rng)

            n_budget_steps: total number of individual budget steps

            data: instance of a MaskedEnvironmentData initialized with Step.ONE

            mab_algorithm: specifices whether the learner should be implemented
                with a Gaussian Process Thompson Sampling or Gaussian Process
                UCB1

            threshold: the minimum average drop of reward that triggers the detection

            threshold_window: integer representing the size of the moving window on
                which the algorithm computes the average reward, which is then
                compared to previous rewards in order to detect a drop in value

            units_less: if set to True the implemented learner will be an
                AlphaUnitslessLearner instance, otherwise it's a
                AlphalessLearner instance

        Returns:
            A ChangeDetectionLearner instance, which adopts the same framework as the
            parent class Learner

        """

        if units_less:
            self.learner = AlphaUnitslessLearner(
                rng, n_budget_steps, data, mab_algorithm
            )
        else:
            self.learner = AlphalessLearner(rng, n_budget_steps, data, mab_algorithm)

        self.threshold_window = threshold_window
        self.threshold = threshold
        self.collected_rewards = list()
        self.t = 0

    def learn(
        self, interactions: List[Interaction], reward: float, prediction: np.ndarray
    ):

        self.t += 1
        self.collected_rewards.append(reward)

        # Start change detection only when there is wnough values to compare
        if self.t >= self.threshold_window * 2:

            window = self.threshold_window

            # The algorithm compares the average reward on the last K*2 values.
            # If the average reward of the older K values is significantly larger
            # than the average reward of the newer K values, a change is detected
            avg_recent_reward = np.mean(self.collected_rewards[-window:])
            avg_previous_reward = np.mean(
                self.collected_rewards[(-2 * window) : -window]
            )

            diff = avg_previous_reward - avg_recent_reward

            # If there is a significant drop in reward, the change is detected and the
            # learner is reset
            if diff > self.threshold:
                logger.debug("Change detected!")
                self.learner.reset()
                self.t = 0

        self.learner.learn(interactions, reward, prediction)

    def predict(
        self, data: MaskedEnvironmentData
    ) -> Tuple[np.ndarray, Optional[List[List[Feature]]]]:
        return self.learner.predict(data)

    def show_progress(self, _):
        pass
