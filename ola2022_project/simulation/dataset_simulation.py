import tqdm
from ola2022_project.environment.environment import (
    create_masked_environment,
    Step,
    Interaction,
)
from collections import namedtuple
from typing import List
import numpy as np

# ------------------------------------------ ATTENTION ------------------------------------------ #
# Temporary and ugly alternative to the classic simulation since it doesn't offer a general
# interface able to train a learner from an existing dataset; most of the code is copied over from
# the original file with some minor modifications in order to adapt it to the context generation
# use case of the simulation.
# ----------------------------------------------------------------------------------------------- #


# Named tuple containing the parameters used by a dataset_simulation
DatasetSimParameters = namedtuple(
    "DatasetSimParameters", ["rng", "env", "learner_factory", "n_budget_steps", "step"]
)


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


def dataset_simulation(
    sim_param,
    dataset,
):

    """Performs a simulation given a learner algorithm and a dataset of interactions
    over a span of days.

    Arguments:
        parameters: parameters used by the simulation to create the environment and the learner

        dataset: list of interactions to analyze

    Returns:
        The collected rewards of running the dataset.

    """

    masked_env = create_masked_environment(sim_param.step, sim_param.env)

    if sim_param.step == Step.ZERO:
        # Creation of clairovyant learner
        learner = sim_param.learner_factory(sim_param.n_budget_steps)

    elif sim_param.step == Step.ONE:
        # Creation of alphaless learner
        learner = sim_param.learner_factory(
            sim_param.rng, sim_param.n_budget_steps, sim_param.env
        )

    collected_rewards = []

    for day_interactions in tqdm(dataset, desc="day"):
        # Ask the learner to estimate the budgets to assign
        budgets = learner.predict(masked_env)

        rewards = _get_aggregated_reward_from_interactions(
            day_interactions, sim_param.env.product_prices
        )

        collected_rewards.append(rewards)

        # Update learner with new observed reward
        learner.learn(rewards, budgets)

    return collected_rewards
