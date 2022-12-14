import numpy as np
from ola2022_project.environment.environment import (
    EnvironmentData,
    alpha_function,
    get_day_of_interactions,
)

from ola2022_project.optimization import budget_assignment


"""WARNING: this module does not work correctly 100% of the time! For some reason, the function
find_optimal_superarm is flawed because in some condition isn't able to find the optimal superarm
but only a close one. - TODO
"""


def clairvoyant_reward(
    env: EnvironmentData,
    population: int,
    max_budget: int,
    n_budget_steps: int,
    custom_superarm=None,
):

    """Computes the average reward playing the optimal superarm according
    to the given parameters. It simulates a deterministic day using the optimal
    superarm.

    Arguments:
        env: instance of EnvironmentData

        population: int representing the total population

        max_budget: maximum budget assignable according to the knapsack constraint

        n_budget_steps: number of budget_steps

    Returns:
        An integer representing the expected reward while playing the optimal superarm.
        This corresponds to the average optimal reward. Because of randomness, this reward
        could be surpassed by a learner sometimes. This is perfectly normal.
    """

    budget_steps = np.linspace(0, max_budget, n_budget_steps)

    if not custom_superarm:
        optimal_superarm = find_optimal_superarm(env, budget_steps)
        optimal_assignment = budget_steps[optimal_superarm]

    else:
        optimal_assignment = custom_superarm

    deterministic_day = get_day_of_interactions(
        np.random.default_rng(),
        population,
        (optimal_assignment, None),
        env,
        deterministic=True,
    )
    units_sold = np.array(
        [interaction.items_bought for interaction in deterministic_day]
    )

    return np.sum(units_sold * env.product_prices) * np.mean([1, env.max_items])


def find_optimal_superarm(
    env: EnvironmentData, budget_steps: np.ndarray, aggregated=True
):

    """Given an environment and a numpy array of budget steps finds the optimal
    superarm to play in such environment given the specified budget steps.

    Arguments:
        env: an instance of EnvironmentData

        budget_steps: numpy array with n_budget_steps + 1 elements, ranging from 0
            to max_budget

        aggregated: if set to False will generate different superarms for each class.
            STILL NOT IMPLEMENTED. - TODO

    Returns:
        An array with P elements (where P is the number of product subcampaigns).
        Every element represents the assigned budget to the i-th subcampaign.
        The total sum amounts to max_budget.
    """

    n_products = len(env.product_prices)
    n_budget_steps = len(budget_steps)
    n_classes = len(env.class_ratios)

    budget_value = np.zeros((n_products, n_budget_steps))
    multipliers = _compute_reward_multiplier(env)

    for i in range(n_products):
        for j, budget in enumerate(budget_steps):
            for user_class in range(n_classes):
                params = env.classes_parameters[user_class][i]
                budget_value[i, j] += alpha_function(
                    budget / n_classes, params.upper_bound, params.max_useful_budget
                )
            budget_value[i, j] *= multipliers[i]
    return budget_assignment(budget_value)


def _compute_reward_multiplier(
    env: EnvironmentData, population=1000, budget=[100, 100, 100, 100, 100]
):

    """Computes reward multipliers for every product. Basically reward multiplier tells
    how much profit a single product will bring on average, according to all the possible
    paths on the graph. The parameters pupolation and budget are dummy parameters, they
    are not relevant and do not influence the result.
    """

    interactions = get_day_of_interactions(
        np.random.default_rng(), population, (budget, None), env, deterministic=True
    )
    interactions_per_product = [
        [elem.items_bought for elem in interactions if elem.landed_on == i]
        for i in range(len(env.product_prices))
    ]

    units_sold_per_product = [
        np.sum(items, axis=0) for items in interactions_per_product
    ]
    n_users_landed_per_product = [len(items) for items in interactions_per_product]

    return [
        np.sum(total_units_sold * env.product_prices, axis=0) / n_users
        for total_units_sold, n_users in zip(
            units_sold_per_product, n_users_landed_per_product
        )
    ]
