from typing import List
import numpy as np
from ola2022_project.environment.environment import (
    EnvironmentData,
    UserClassParameters,
    alpha_function,
)

from ola2022_project.optimization import budget_assignment


def find_optimal_arms(
    env: EnvironmentData, budget_steps: np.ndarray, population: int, aggregated=True
):

    n_products = len(env.product_prices)
    n_budget_steps = len(budget_steps)
    n_classes = len(env.class_ratios)

    budget_value = np.array([])

    for user_class_parameters, ratio in zip(env.classes_parameters, env.class_ratios):

        class_budget_value_matrix = np.array([])
        class_multipliers = compute_reward_multiplier(env, user_class_parameters)

        for product_paramters in user_class_parameters:

            for step in budget_steps:
                class_budget_value_matrix = np.append(
                    class_budget_value_matrix,
                    alpha_function(
                        step,
                        product_paramters.upper_bound,
                        product_paramters.max_useful_budget,
                    )
                    * ratio
                    * population,
                )

        class_budget_value_matrix = np.reshape(
            class_budget_value_matrix, (n_products, n_budget_steps)
        )
        class_budget_value_matrix = (
            class_budget_value_matrix * np.atleast_2d(class_multipliers).T
        )

        np.append(budget_value, class_budget_value_matrix)

    if aggregated:
        budget_value = np.sum(budget_value, axis=0)

    else:
        budget_value = np.reshape(
            budget_value, (n_products * n_classes, n_budget_steps)
        )

    return budget_assignment(budget_value)


def compute_reward_multiplier(
    env: EnvironmentData, class_parameters: List[UserClassParameters]
):

    prices = np.array(env.product_prices)
    n_products = len(env.product_prices)
    reservation_prices = [product.reservation_price for product in class_parameters]

    reward_multiplier = list()

    for product in range(n_products):

        conversion_rates = np.zeros(n_products)

        conversion_rates = _simulate_interaction(
            product, env, reservation_prices, conversion_rates, 1
        )

        reward = np.sum(prices * np.array(conversion_rates))
        reward_multiplier.append(reward)

    return reward_multiplier


def _simulate_interaction(
    current_product,
    classes_left,
    env: EnvironmentData,
    reservation_prices,
    conversion_rates,
    graph_value,
):
    if reservation_prices[current_product] >= reservation_prices[current_product]:
        return conversion_rates

    conversion_rates[current_product] = graph_value

    slot1, slot2 = env.next_products[current_product]

    if conversion_rates[slot1] == 0:
        conversion_rates = _simulate_interaction(
            slot1,
            classes_left,
            env,
            reservation_prices,
            conversion_rates,
            conversion_rates[current_product] * env.graph[current_product, slot1],
        )

    if conversion_rates[slot2] == 0:
        conversion_rates = _simulate_interaction(
            slot2,
            classes_left,
            env,
            reservation_prices,
            conversion_rates,
            conversion_rates[current_product]
            * env.lam
            * env.graph[current_product, slot2],
        )

    return conversion_rates
