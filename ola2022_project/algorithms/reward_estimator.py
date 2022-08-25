import numpy as np
from ola2022_project.environment.environment import MaskedEnvironmentData


def compute_user_influence(unaffordable_ratio, env: MaskedEnvironmentData):

    affordable_ratio = 1 - unaffordable_ratio
    estimated_reward = list()
    n_products = len(env.product_prices)

    for p in range(n_products):

        potential_sold_units = np.zeros(n_products)
        bought = [False for _ in range(len(env.product_prices))]

        potential_sold_units[p] = affordable_ratio[p]

        bought[p] = True

        if potential_sold_units[p] == 1:
            selected_population = affordable_ratio
        else:
            selected_population = np.ones(5)

        estimated_reward.append(
            _simulate_interaction(
                p,
                env.next_products[p],
                bought,
                potential_sold_units,
                selected_population,
                env,
            )
        )

    return np.array(estimated_reward) @ np.array(env.product_prices)


def _simulate_interaction(
    curr_prod,
    next_prod,
    bought,
    potential_sold_units,
    selected_population,
    env: MaskedEnvironmentData,
):
    bought[curr_prod] = True

    slot1, slot2 = next_prod

    if not bought[slot1]:
        potential_sold_units[slot1] = (
            potential_sold_units[curr_prod]
            * env.graph[curr_prod, slot1]
            * selected_population
        )
        _simulate_interaction(
            slot1,
            env.next_products[slot1],
            bought,
            potential_sold_units,
            selected_population,
            env,
        )

    if not bought[slot2]:
        potential_sold_units[slot2] = (
            potential_sold_units[curr_prod]
            * env.graph[curr_prod, slot2]
            * selected_population
            * env.lam
        )
        _simulate_interaction(
            slot2,
            env.next_products[slot2],
            bought,
            potential_sold_units,
            selected_population,
            env,
        )

    return potential_sold_units
