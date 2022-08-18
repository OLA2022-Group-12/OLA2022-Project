import enum
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck


class BaseMAB:

    """Base Multi-Armed Bandit class that will be implemented by the actual
    bandit algorithms. It doesn't really help avoiding much code repetition, so
    it's a bit useless, but it is probably more clear for anyone that followed
    the course's exercise sessions
    """

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])

    def _update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)


class GPTSLearner(BaseMAB):

    """Implementation of the Gaussian Process Tomphson Sampling algorithm. Used
    mainly to fit and estimate a function using a gaussian process and normal
    distributions
    """

    def __init__(
        self,
        rng,
        n_arms,
        arms,
        std=10,
        kernel_range=(1e-2, 1e4),
        kernel_scale=1,
        theta=1.0,
        l_param=10,
    ):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.ones(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * std
        self.pulled_arms = list()
        self.alpha = std
        self.rng = rng
        self.kernel = (
            Ck(theta, kernel_range) * RBF(l_param, kernel_range) * kernel_scale
        )
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, alpha=self.alpha**2, n_restarts_optimizer=2
        )

    def _update_observations(self, arm_idx, reward):
        super()._update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def _update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(
            np.atleast_2d(self.arms).T, return_std=True
        )
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):

        """This function must be called when a reward is observed after an arm
        is pulled. It updates the multi armed bandit's internal estimation using
        the new collected value.

        Arguments:
            pulled_arm: the index of the previously pulled arm (not the value itself)
                in the budget steps array

            reward: the reward observed after pulling the arm of the previous
                argument
        """

        self.t += 1
        self._update_observations(pulled_arm, reward)
        self._update_model()

    def estimation(self):

        """Returns an estimation of the reward for every budget step.

        Returns: a numpy array with n_budget_steps elements containing the
        estimated reward for every step
        """
        return self.rng.normal(self.means, self.sigmas)


class Mab(enum.Enum):
    GPTS = ()
    GPUCB1 = ()
