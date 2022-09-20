import logging
from ola2022_project.learners.learner import (
    ClairvoyantLearner,
    StupidLearner,
    AlphalessLearner,
    AlphaUnitslessLearner,
    GraphlessLearner,
)
from tqdm.notebook import trange
from ola2022_project.environment.environment import (
    get_day_of_interactions,
    create_masked_environment,
    Step,
    Interaction,
    EnvironmentData,
)
from typing import List
import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class Simulation:

    """Class used for instantiating and running an interactive simulation
    that can be customized and edited between different executions.
    The outputs of the simulation are stored in the 'dataset' and 'rewards' attributes.
    """

    def __init__(
        self,
        rng: Generator,
        env: EnvironmentData,
        step: Step = Step.ZERO,
        n_budget_steps: int = 5,
        n_customers_mean: int = 100,
        n_customers_variance: int = 10,
        **learner_params,
    ):

        """Constructor for the Simulation.

        Arguments:
            rng: randomness generator

            env: environment where the simulation is going to be run

            step: step number of the simulation, related to the various steps requested
            by the project specification and corresponding to which properties
            of the environment are masked to the learner and to which learner is going
            to be instantiatiated

            n_budget_steps: number of steps in which the budget must be divided

            n_customers_mean: expected value of the number of new potential
            customers every day

            n_customers_variance: variance of the daily number of potential customers

            learner_params: various parameters used to created the selected learner
        """

        self.rng = rng
        self.env = env
        self.step = step
        self.n_budget_steps = n_budget_steps
        self.n_customers_mean = n_customers_mean
        self.n_customers_variance = n_customers_variance

        self._masked_env = create_masked_environment(self.env)

        self.reset(True, learner_params)

    @property
    def step(self):
        return self.step

    @step.setter
    def step(self, value: Step):
        self.step = value
        self.learner = self._learner_init()

    @property
    def env(self):
        return self.env

    @env.setter
    def env(self, value: EnvironmentData):
        self.env = value
        self._masked_env = create_masked_environment(self.env)

    def _learner_init(self, **params):

        """Creates a new learner utilizing the current simulation step and the
        given learner creation parameters.

        Arguments:
            params: learner creation parameters

        Returns:
            A new untrained learner depending on the current simulation step
        """

        if self.step == Step.CLAIRVOYANT:
            # Creation of clairovyant learner
            return ClairvoyantLearner(self.n_budget_steps)
        if self.step == Step.ZERO:
            # Creation of stupid learner
            return StupidLearner()
        elif self.step == Step.ONE:
            # Creation of alphaless learner
            return AlphalessLearner(
                self.rng,
                self.n_budget_steps,
                self.masked_env,
                mab_algorithm=params.mab_algorithm,
            )
        elif self.step == Step.TWO:
            # Creation of alphaunitsless learner
            return AlphaUnitslessLearner(
                self.rng,
                self.n_budget_steps,
                self.masked_env,
                mab_algorithm=params.mab_algorithm,
            )
        elif self.step == Step.THREE:
            # Creation of graphless learner
            return GraphlessLearner(self.rng, self.n_budget_masked_env)
        else:
            raise NotImplementedError(f"cannot handle step {self.step} yet")

    def simulate(
        self,
        n_days: int = 100,
        show_progress_graphs: bool = False,
    ):

        """Simulates a given number of days of the simulation while appending all the
        results to the dedicated simulation attributes.

        Arguments:
            n_days: number of days to run the simulation for

            show_progress_graphs: if set to True, will visualize the learner progress graphs
            (if implemented) at each iteration
        """

        for day in trange(n_days, desc="days"):
            # Every day, there is a number of new potential customers drawn
            # from a normal distribution, rounded to the closest integer
            n_new_customers = int(
                np.rint(
                    self.rng.normal(self.n_customers_mean, self.n_customers_variance)
                )
            )

            # The mimnum number of customers is set to 1, so that none of the
            # operations of the environment requiring division computation break
            # and we avoid inconsistent data like a negative number of customers
            if n_new_customers <= 0:
                n_new_customers = 1

            logger.debug(f"Got {n_new_customers} new customer(s) on day {day}")

            # Ask the learner to estimate the budgets to assign
            budgets = self.learner.predict(self.masked_env)

            # Compute interactions for the entire day
            interactions = get_day_of_interactions(
                self.rng, n_new_customers, budgets, self.env
            )
            self.dataset.append(interactions)
            logger.debug(f"Interactions: {interactions}")

            # Compute rewards from interactions
            rewards = self._get_aggregated_reward_from_interactions(interactions)
            self.rewards.append(rewards)
            logger.debug(f"Rewards: {rewards}")

            # Update learner with new observed reward
            self.learner.learn(interactions, rewards, budgets)

            if show_progress_graphs:
                fig = plt.figure()
                self.learner.show_progress(fig)
                plt.show(block=True)

        self.tot_days += n_days

    def simulate_n(
        self,
        n_days: int = 100,
        n_experiments=2,
        show_progress_graphs: bool = False,
    ):
        # TODO: remove and create external wrapper
        pass

    def _get_aggregated_reward_from_interactions(self, interactions: List[Interaction]):

        """Computes the margin made each day, for each of the 3 classes of users.

        Arguments:
            interactions: A list of tuples, with a size corresponding to num_customers. Every
            tuple denotes an interaction. tuple[0] is the user's class, which can be
            1, 2 or 3. tuple[1] is a numpy array of 5 elemets, where every element i
            represents how many of the products i+1 the customer bought

            prices: Price of the 5 products

        Returns:
            An integer representing the total aggregated reward of the entire list of interactions.
        """

        # Creates a list contaninig only the number of units every customer bought
        # for each product
        units_sold = np.array([i.items_bought for i in interactions])

        # First with np.sum() we compute a single array containing how many units we
        # sold for every product. Then the units are multiplied element-wise by the
        # price of the corresponding product
        reward_per_product = (
            np.sum(units_sold, axis=0) * self.env.product_prices
            if len(units_sold) > 0
            else np.array([])
        )

        # The profit of all the products are summed
        return np.sum(reward_per_product)

    def reset(self, reset_learner: bool = False, **learner_params):

        """Resets the dynamic parameters of the simulation to their initial values.

        Arguments:
            reset_learner: if set to True, will also reset the learner by creating a new one
            utilizing the current simulation step as a reference

            learner_params: if the reset_learner flag is set to True, the learner_params will
            be passed to the new learner that will be created
        """

        self.tot_days = 0
        self.dataset = np.array([])
        self.rewards = np.array([])
        if reset_learner:
            self.learner = self._learner_init()
