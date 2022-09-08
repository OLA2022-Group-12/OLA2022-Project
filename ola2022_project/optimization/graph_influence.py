from typing import List, Tuple

from ola2022_project.environment.environment import EnvironmentData


def get_all_interaction_from_node(
    primary_product: int, user_class: int, env_data: EnvironmentData
) -> float:
    """
    Calculates the average price of purchases
    taking into account all possible paths starting from a node.

    Arguments:
      primary_product: integer representing the primary product (starting node)

      user_class: integer representing user's class

      env_data: instance of EnvironmentData

    Returns:
      A float representing the average money spent by a customer starting from a node

    """

    history = [[primary_product]]
    newhistory = []

    price_history = [[env_data.product_prices[primary_product]]]
    newprice_history = []

    # The algorithm is repeated until all possibilities have been explored.
    # We use a condition in the loop to exit this infinite loop.
    # We check if the history is the same as in the previous round,
    # which means that all paths have been explored.
    while True:
        newhistory = []
        newprice_history = []

        # For each path, we see if we can extend them by
        # looking if there are still secondary products possible.
        for i in range(len(history)):
            next_, expected_income = get_next_product(history[i], user_class, env_data)

            # This particular path cannot be extended
            if len(next_) == 0:
                newhistory.append(history[i])
                newprice_history.append(price_history[i])

            # If there are secondary products available, we add them in the history
            for k in range(len(next_)):
                h = history[i] + [next_[k]]
                newhistory.append(h)

                hp = price_history[i] + [expected_income[k]]
                newprice_history.append(hp)

        # We check if the history is the same as in the previous round,
        # which means that all paths have been explored.
        if newhistory == history:
            break
        else:
            history = newhistory
            price_history = newprice_history
    reward = [sum(i) for i in newprice_history]

    return sum(reward) / len(reward)


def get_next_product(
    current_path: List[int], user_class: int, env_data: EnvironmentData
) -> Tuple[List[int], List[int]]:

    """
    Return the secondary products possible,
    depending on if the budget allows it and if it has not already been bought,
    as well as the price of the purchases weighted by the probability according
    to the graph and lambda

    Arguments:
      current_path: List containing products already displayed as primary product.

      user_class: integer representing user's class

      env_data: instance of EnvironmentData

    Returns:
      Tuple of Lists :
        The secondary products possible
        The expected price associated for this products

    """

    primary_product = current_path[-1]
    secondary_products = env_data.next_products[primary_product]
    secondary_prices = [
        env_data.product_prices[secondary_products[0]],
        env_data.product_prices[secondary_products[1]],
    ]
    user_reservation_prices = [
        env_data.classes_parameters[user_class][
            secondary_products[0]
        ].reservation_price,
        env_data.classes_parameters[user_class][
            secondary_products[1]
        ].reservation_price,
    ]

    next_ = []
    expected_income = []

    # Checks if the price of the first secondary product is under the reservation
    # price of the user class
    # And that the product has not already been displayed as a primary product
    if (
        secondary_prices[0] < user_reservation_prices[0]
        and not secondary_products[0] in current_path
    ):
        next_.append(secondary_products[0])
        expected_income.append(
            secondary_prices[0] * env_data.graph[primary_product, secondary_products[0]]
        )

    # Checks if the price of the second secondary product is under the reservation
    # price of the user class
    # And that the product has not already been displayed as a primary product
    if (
        secondary_prices[1] < user_reservation_prices[1]
        and not secondary_products[1] in current_path
    ):
        next_.append(secondary_products[1])
        expected_income.append(
            secondary_prices[1]
            * env_data.graph[primary_product, secondary_products[1]]
            * env_data.lam  # NB! This is the difference between the two ifs
        )

    return (next_, expected_income)


def get_expected_value_per_node(
    user_class: int, env_data: EnvironmentData
) -> List[float]:
    """
    Calculates the expected money per node

    Arguments:
        user_class: integer representing user's class
        env_data: instance of Environment_data

    Returns:
        A list with expected money spent by a user starting from node i
    """
    reward_per_seed = [
        get_all_interaction_from_node(seed, user_class, env_data)
        for seed, _ in enumerate(env_data.product_prices)
    ]
    return reward_per_seed
