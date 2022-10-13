import numpy as np
from ola2022_project.environment.environment import (
    EnvironmentData,
    alpha_function,
    get_day_of_interactions,
)

from ola2022_project.optimization import budget_assignment


def find_optimal_superarm(
    env: EnvironmentData, budget_steps: np.ndarray, population: int, aggregated=True
):

    n_products = len(env.product_prices)
    n_budget_steps = len(budget_steps)
    n_classes = len(env.class_ratios)

    budget_value = np.zeros((n_products, n_budget_steps))
    multipliers = compute_reward_multiplier(env)

    for i in range(n_products):
        for j, budget in enumerate(budget_steps):
            for user_class in range(n_classes):
                params = env.classes_parameters[user_class][i]
                budget_value[i, j] += (
                    alpha_function(budget, params.upper_bound, params.max_useful_budget)
                    * multipliers[i]
                )

    return budget_assignment(budget_value)


def compute_reward_multiplier(
    env: EnvironmentData, population=1000, budget=[100, 100, 100, 100, 100]
):

    interactions = get_day_of_interactions(
        np.random.default_rng(), population, budget, env, deterministic=True
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
