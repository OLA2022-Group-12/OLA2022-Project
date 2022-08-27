from tqdm.notebook import trange
from ola2022_project.environment.environment import (
    get_day_of_interactions,
    create_masked_environment,
    Step,
    Interaction,
)
from typing import List
import numpy as np


def _get_aggregated_reward_from_interactions(interactions: List[Interaction], prices):

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
    units_sold = [i.items_bought for i in interactions]

    # First with np.sum() we compute a single array containing how many units we
    # sold for every product. Then the units are multiplied element-wise by the
    # price of the corresponding product
    reward_per_product = np.sum(units_sold, axis=0) * prices

    # The profit of all the products are summed
    return np.sum(reward_per_product)


def simulation(
    rng,
    env,
    learner_factory,
    n_customers_mean=100,
    n_customers_variance=10,
    n_experiment=1,
    n_days=100,
    n_budget_steps=5,
    step=Step.ZERO,
):

    """Runs the simulation for a certain amount of experiments consisting of a
    certain amount of days

    Arguments:
        env: an instance of the Environment class

        learner_factory: a function which creates a new learner (needed to run
        multiple experiments with "fresh" learners)

        n_customers_mean: expected value of the number of new potential
            customers every day

        n_customers_variance: variance of the daily number of potential
        customers

        n_experiment: number of times the experiment is performed,
          to have statistically more accurate results. By default, the value is
          1 because in the real world we don't have time to do each experiment
          several times.

        n_days: duration of the experiment in days

        n_budget_steps: number of steps in which the budget must be divided

        step: Step number of the simulation, related to the various steps
          requested by the project specification and corresponding to which
          properties of the environment are masked to the learner

    Returns:
        The collected rewards of running each experiment

    """

    rewards_per_experiment = []

    masked_env = create_masked_environment(step, env)

    for _ in trange(n_experiment, desc="experiment"):

        if step == Step.ZERO:
            # Creation of clairovyant learner
            learner = learner_factory(n_budget_steps)

        elif step == Step.ONE:
            # Creation of alphaless learner
            learner = learner_factory(rng, n_budget_steps, masked_env)

        collected_rewards = []

        for _ in trange(n_days, desc="day"):
            # Every day, there is a number of new potential customers drawn from a
            # normal distribution, rounded to the closest integer
            n_new_customers = int(
                np.rint(rng.normal(n_customers_mean, n_customers_variance))
            )

            # The mimnum number of customers is set to 1, so that none of the
            # operations of the environment requiring division computation break
            # and we avoid not consistent datab like a negative number of customers
            if n_new_customers <= 0:
                n_new_customers = 1

            # Ask the learner to estimate the budgets to assign
            budgets = learner.predict(masked_env)

            # Compute interactions for the entire day
            interactions = get_day_of_interactions(rng, n_new_customers, budgets, env)

            rewards = _get_aggregated_reward_from_interactions(
                interactions, env.product_prices
            )

            collected_rewards.append(rewards)

            # Update learner with new observed reward
            learner.learn(rewards, budgets)

        rewards_per_experiment.append(collected_rewards)

    return rewards_per_experiment
