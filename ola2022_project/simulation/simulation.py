import logging
import copy
import numpy as np
import matplotlib.pyplot as plt
from ola2022_project.learners import (
    ClairvoyantLearner,
    StupidLearner,
    AlphalessLearner,
    AlphaUnitslessLearner,
    GraphlessLearner,
    SlidingWindowLearner,
    ChangeDetectionLearner,
)
from tqdm.notebook import trange
from ola2022_project.environment.environment import (
    get_day_of_interactions,
    create_masked_environment,
    feature_filter,
    Step,
    Interaction,
    EnvironmentData,
    Feature,
)
from typing import List
from numpy.random import Generator


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
        n_budget_steps: int = 20,
        population_mean: int = 100,
        population_variance: int = 10,
        include_learner: bool = True,
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

            population_mean: expected value of the number of new potential
            customers every day

            population_variance: variance of the daily number of potential customers

            include_learner: if false, don't create a learner with the simulation

            learner_params: various parameters used to created the selected learner
        """

        self.rng = rng
        self._step = step
        self.env = (
            env  # Exploit the property setter to also create the masked environment
        )
        self.n_budget_steps = n_budget_steps
        self.population_mean = population_mean
        self.population_variance = population_variance
        self.learner_params = learner_params

        self.reset(include_learner)

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value: Step):
        self._step = value
        self.learner = self._learner_init()

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, value: EnvironmentData):
        self._env = value
        self.masked_env = create_masked_environment(self.step, self.env)

    def _learner_init(self):

        """Creates a new learner utilizing the current simulation step and the
        given learner creation parameters.

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
                mab_algorithm=self.learner_params["mab_algorithm"],
            )

        elif self.step == Step.TWO:
            # Creation of alphaunitsless learner
            return AlphaUnitslessLearner(
                self.rng,
                self.n_budget_steps,
                self.masked_env,
                mab_algorithm=self.learner_params["mab_algorithm"],
            )

        elif self.step == Step.THREE:
            # Creation of graphless learner
            return GraphlessLearner(self.rng, self.n_budget_steps, self.masked_env)

        elif self.step == Step.FOUR:

            if self.learner_params["sliding_window"]:
                return SlidingWindowLearner(
                    self.rng,
                    self.n_budget_steps,
                    self.masked_env,
                    mab_algorithm=self.learner_params.get("mab_algorithm"),
                    window_size=self.learner_params.get("window_size"),
                    units_less=self.learner_params.get("units_less"),
                )

            else:
                return ChangeDetectionLearner(
                    self.rng,
                    self.n_budget_steps,
                    self.masked_env,
                    mab_algorithm=self.learner_params["mab_algorithm"],
                    threshold=self.learner_params["threshold"],
                    threshold_window=self.learner_params["threshold_window"],
                    units_less=self.learner_params["units_less"],
                )

        elif self.step == Step.FIVE:
            # Creation of contextual learner
            # Workaround for circular imports
            from ola2022_project.learners.contextual_learner import ContextualLearner

            return ContextualLearner(
                self.rng,
                self.n_budget_steps,
                self.masked_env,
                simulation=self,
                features=self.learner_params["features"],
                mab_algorithm=self.learner_params["mab_algorithm"],
            )
        else:
            raise NotImplementedError(f"cannot handle step {self.step} yet")

    def simulate(
        self,
        n_days: int = 100,
        features: List[Feature] = [],
        update: bool = True,
        show_progress_graphs: bool = False,
    ):

        """Simulates a given number of days of the simulation while appending all the
        results to the dedicated simulation attributes.

        Arguments:
            n_days: number of days to run the simulation for

            features: list of features that will be used to filter the users in order to have
            a specialized training dataset; if empty, no filtering will be done

            update: flag to decide whether to update or not the current learner

            show_progress_graphs: if set to True, will visualize the learner progress graphs
            (if implemented) at each iteration
        """

        for day in trange(n_days, desc="days"):
            # Every day, there is a number of new potential customers drawn
            # from a normal distribution, rounded to the closest integer
            population = int(
                np.rint(self.rng.normal(self.population_mean, self.population_variance))
            )

            # The mimnum number of customers is set to 1, so that none of the
            # operations of the environment requiring division computation break
            # and we avoid inconsistent data like a negative number of customers
            if population <= 0:
                population = 1

            logger.debug(f"Got {population} new customer(s) on day {day}")

            # Ask the learner to estimate the budgets to assign
            budgets = self.learner.predict(self.masked_env)

            # Compute interactions for the entire day
            interactions = get_day_of_interactions(
                self.rng, population, budgets, self.env
            )
            self.dataset.append(interactions)

            # Filter interactions based on features
            if features:
                interactions = feature_filter([interactions], features)[0]
                self.filtered_dataset.append(interactions)
            logger.debug(f"Interactions: {interactions}")

            # Compute rewards from interactions
            rewards = _get_aggregated_reward_from_interactions(
                self.env.product_prices, interactions
            )
            self.rewards = np.append(self.rewards, rewards)
            logger.debug(f"Rewards: {rewards}")

            # Update learner with new observed reward
            if update:
                self.learner.learn(interactions, rewards, budgets[0])

            if show_progress_graphs:
                fig = plt.figure()
                self.learner.show_progress(fig)
                plt.show(block=True)

        self.tot_days += n_days

    def reset(self, reset_learner: bool = False):

        """Resets the dynamic parameters of the simulation to their initial values.

        Arguments:
            reset_learner: if set to True, will also reset the learner by creating a new one
            utilizing the current simulation step as a reference
        """

        self.tot_days = 0
        self.dataset = []
        self.filtered_dataset = []
        self.rewards = np.array([])
        if reset_learner:
            self.learner = self._learner_init()

    def copy(self, include_learner: bool = True):

        """Generates a new Simulation copying the parameters of the current simulation

        Arguments:
            include_learner: if false, don't include a learner in the simulation (note that if
            this flag is true the learner of the target simulation won't be copied over but a new
            one will be created)

        Returns:
            a new copied Simulation object
        """

        return Simulation(
            self.rng,
            copy.copy(self.env),
            self.step,
            self.n_budget_steps,
            self.population_mean,
            self.population_variance,
            include_learner,
            **self.learner_params,
        )


def _get_aggregated_reward_from_interactions(
    product_prices, interactions: List[Interaction]
):

    """Computes the margin made each day, for each of the 3 classes of users.

    Arguments:
        product_prices: Reference product prices used to compute the reward

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
        np.sum(units_sold, axis=0) * product_prices
        if len(units_sold) > 0
        else np.array([])
    )

    # The profits of all the products are summed
    return np.sum(reward_per_product)


def create_n(
    rng: Generator,
    env: EnvironmentData,
    step: Step = Step.ZERO,
    n: int = 2,
    n_budget_steps: int = 20,
    population_mean: int = 100,
    population_variance: int = 10,
    **learner_params,
):

    """Helper function that simplifies the creation of multiple equal simulations at once

    Arguments:
        n: number of simulations to create

        rng: randomness generator

        env: environment where the simulation is going to be run

        step: step number of the simulation, related to the various steps requested
        by the project specification and corresponding to which properties
        of the environment are masked to the learner and to which learner is going
        to be instantiatiated

        n_budget_steps: number of steps in which the budget must be divided

        population_mean: expected value of the number of new potential
        customers every day

        population_variance: variance of the daily number of potential customers

        learner_params: various parameters used to created the selected learner

    Returns:
        A list of new simulations with the parameters specified

    """

    return [
        Simulation(
            rng,
            env,
            step=step,
            n_budget_steps=n_budget_steps,
            population_mean=population_mean,
            population_variance=population_variance,
            **learner_params,
        )
        for i in range(n)
    ]


def simulate_n(
    simulations: List[Simulation],
    n_days: int = 100,
    show_progress_graphs: bool = False,
):

    """Helper function that simplifies the act of running multiple simulations with the purpose
    of visualization of the results

    Arguments:
        simulations: list of simulation objects that will be run

        n_days: number of days to run each simulation for

        show_progress_graphs: if set to True, will visualize the learner progress graphs
        (if implemented) at each iteration

    Returns:
        A list containing all the collected rewards obtained by running all the simulations.

    """

    rewards = []

    for sim in simulations:
        sim.simulate(n_days, show_progress_graphs)
        rewards.append(sim.rewards)

    return rewards
