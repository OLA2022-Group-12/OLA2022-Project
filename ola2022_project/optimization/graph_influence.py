import logging
from typing import List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def get_influence_of_seed(
    graph: NDArray[np.float64],
    seed: int,
    graph_squared: Optional[NDArray[np.float64]] = None,
    graph_cubed: Optional[NDArray[np.float64]] = None,
) -> float:
    """Get the influence of a node in a graph, i.e. the "amount" of nodes that
    the graph will activate when the adjacency matrix contains the activation
    weight/probability.

    This function is specialized for the following criterions:
        - Only consider neighbors within three degress of separation from the
          seed
        - A node can only be activated once
        - A node cannot activate itself

    Arguments:
        graph: numpy matrix (NxN) with floats between 0.0 and 1.0 which
        determine the weight/probability of activation

        seed: integer between 0 and N-1 which determines the node to calculate
        influence from

        graph_squared: optionally provide numpy matrix (NxN) of the graph
        squared, matrix above. This can be used as an optimization if the graph
        is explored from several seeds.

        graph_cubed: optionally provide numpy matrix (NxN) of the graph cubed,
        matrix above. This can be used as an optimization if the graph is
        explored from several seeds.
    """
    # Store a vector with 1 where a node is active (so only seed for now)
    node_active = np.array([int(i == seed) for i in range(len(graph))])

    # Probability to activate direct neighbors (not including seed)
    direct_prob = graph[seed] * (1 - node_active)

    # Update active nodes with the source nodes of the last step
    node_active += np.where(direct_prob > 0.0, 1, 0)

    # Probability to activate neighbors of direct neighbors.
    # After taking the matrix power, the row of our seed is the probability to
    # reach the node j from the start node. We multiply by (1 - node_active) to
    # ensure that already active nodes don't contribute more than once.
    second_graph = np.linalg.matrix_power(graph, 2) if graph_squared is None else graph_squared
    second_indirect_prob = second_graph[seed] * (1 - node_active)

    # Update active nodes with the source nodes of the last step
    node_active += np.where(second_indirect_prob > 0.0, 1, 0)

    # Probability to activate neighbors of neighbors of neighbors
    third_graph = np.matmul(graph, second_graph) if graph_cubed is None else graph_cubed
    third_indirect_prob = third_graph[seed] * (1 - node_active)

    logger.debug(f"first_prop  = {direct_prob}")
    logger.debug(f"second_prop = {second_indirect_prob}")
    logger.debug(f"third_prop  = {third_indirect_prob}")

    logger.debug(f"first_graph = {graph}")
    logger.debug(f"second_graph = {second_graph}")
    logger.debug(f"third_graph = {third_graph}")

    return (
        np.sum(direct_prob)
        + np.sum(second_indirect_prob)
        + np.sum(third_indirect_prob)
    )


def get_all_interaction_from_node(
    primary_product: int,
    graph: NDArray[np.float32],
    product_prices: List[float],
    reservation_prices: List[float],
    next_products: List[Tuple[int, int]],
    lam: float,
) -> float:
    """
    Calculates the average price of purchases
    taking into account all possible paths starting from a node.

    Arguments:
      primary_product: integer representing the primary product (starting node)

      graph: an (NxN) matrix which represents the weights between the product nodes in the graph

      product_prices: a list of N product prices of the given products

      reservation_prices: a list of N reservation prices which determine wether
                          a product is bought

      next_products: a list of N secondary products to a product

      lam: the probability multiplier of the second slot

    Returns:
      A float representing the average money spent by a customer starting from a node

    """

    history = [[primary_product]]
    newhistory = []

    price_history = [[product_prices[primary_product]]]
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
            next_, expected_income = get_next_product(
                history[i],
                graph,
                product_prices,
                reservation_prices,
                next_products,
                lam,
            )

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
    current_path: List[int],
    graph: NDArray[np.float32],
    product_prices: List[float],
    reservation_prices: List[float],
    next_products: List[Tuple[int, int]],
    lam: float,
) -> Tuple[List[int], List[int]]:

    """
    Return the secondary products possible,
    depending on if the budget allows it and if it has not already been bought,
    as well as the price of the purchases weighted by the probability according
    to the graph and lambda

    Arguments:
      current_path: List containing products already displayed as primary product.

      graph: an (NxN) matrix which represents the weights between the product nodes in the graph

      product_prices: a list of N product prices of the given products

      reservation_prices: a list of N reservation prices which determine wether
                          a product is bought

      next_products: a list of N secondary products to a product

      lam: the probability multiplier of the second slot

    Returns:
      Tuple of Lists :
        The secondary products possible
        The expected price associated for this products

    """

    primary_product = current_path[-1]
    secondary_products = next_products[primary_product]
    secondary_prices = [
        product_prices[secondary_products[0]],
        product_prices[secondary_products[1]],
    ]
    user_reservation_prices = [
        reservation_prices[secondary_products[0]],
        reservation_prices[secondary_products[1]],
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
            secondary_prices[0] * graph[primary_product, secondary_products[0]]
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
            * graph[primary_product, secondary_products[1]]
            * lam  # NB! This is the difference between the two ifs
        )

    return (next_, expected_income)


def get_expected_value_per_node(
    graph: NDArray[np.float32],
    product_prices: List[float],
    reservation_prices: List[float],
    next_products: List[Tuple[int, int]],
    lam: float,
) -> List[float]:
    """
    Calculates the expected money per node

    Arguments:
      graph: an (NxN) matrix which represents the weights between the product nodes in the graph

      product_prices: a list of N product prices of the given products

      reservation_prices: a list of N reservation prices which determine wether
                          a product is bought

      next_products: a list of N secondary products to a product

      lam: the probability multiplier of the second slot

    Returns:
        A list with expected money spent by a user starting from node i
    """
    reward_per_seed = [
        get_all_interaction_from_node(
            seed,
            graph,
            product_prices,
            reservation_prices,
            next_products,
            lam,
        )
        for seed, _ in enumerate(product_prices)
    ]
    return reward_per_seed
