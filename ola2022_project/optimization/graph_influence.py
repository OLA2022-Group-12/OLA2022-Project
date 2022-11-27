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
) -> Tuple[float, NDArray[np.float64]]:
    """Get the influence of a node in a graph, i.e. the "amount" of nodes that
    the graph will activate when the adjacency matrix contains the activation
    weight/probability.

    This function is specialized for the following criterions:
        - Only considers neighbors up to three degress of separation from the
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

    Returns:
        A tuple which contains the amount of nodes which will be influenced
        (i.e. the average amount of nodes which will be activated) and a array
        of the probability that node i will be activated from the seed.
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
    second_graph = (
        np.linalg.matrix_power(graph, 2) if graph_squared is None else graph_squared
    )
    second_indirect_prob = second_graph[seed] * (1 - node_active)

    # Update active nodes with the source nodes of the last step
    node_active += np.where(second_indirect_prob > 0.0, 1, 0)

    # Probability to activate neighbors of neighbors of neighbors
    third_graph = np.matmul(graph, second_graph) if graph_cubed is None else graph_cubed
    third_indirect_prob = third_graph[seed] * (1 - node_active)

    probs = direct_prob + second_indirect_prob + third_indirect_prob

    logger.debug(f"first_prop  = {direct_prob}")
    logger.debug(f"second_prop = {second_indirect_prob}")
    logger.debug(f"third_prop  = {third_indirect_prob}")
    logger.debug(f"props       = {probs}")

    logger.debug(f"first_graph = {graph}")
    logger.debug(f"second_graph = {second_graph}")
    logger.debug(f"third_graph = {third_graph}")

    return np.sum(probs), probs


def make_influence_graph(
    primary_product: int,
    graph: NDArray[np.float64],
    product_prices: List[float],
    reservation_prices: List[float],
    next_products: List[Tuple[int, int]],
    lam: float,
) -> NDArray[np.float64]:
    """Convert the purely click-ratio graph into an actual influence graph,
    taking slot and reservation price into account. The returned influence graph
    can be used with the `get_influence_of_seed`-function to get the estimated
    value of a product in the graph.

    Arguments:
        primary_product: int which is the product to generate the influence graph for

        graph: a numpy matrix (NxN) which contains the click-ratio to continue
        clicking on a secondary product in the product graph

        product_prices: list of float which is the prices of the product i

        reservation_prices: list of float which is the reservation_price of product i

        next_products: list of tuple of product indicies

        lam: float which determines the lambda factor of secondary slot

    Returns:
        A numpy matrix (NxN) which symolizes the graph of influence between the
        products, i.e. what is the probability of buying product j when you are
        at product i's page.
    """
    influence_graph = np.zeros_like(graph)

    if product_prices[primary_product] > reservation_prices[primary_product]:
        # We will never buy anything cause we cannot buy even the primary
        # product, so return empty (zeroed) graph
        return influence_graph

    # Add chance of clicking neighboring products ONLY IF they will be bought
    # when clicking
    for i, direct_neighbor in enumerate(next_products[primary_product]):
        if product_prices[direct_neighbor] > reservation_prices[direct_neighbor]:
            # We will not buy this neighbor product, so we won't buy any later
            # in the chain neither.
            continue

        # Assign lower influences to the influence graph first, so we will then
        # automatically overwrite if there exists multiple paths to a single
        # product. I.e 1 -> 2 -> 3, 1 -> 3, the latter probability will be used
        # instead of former.
        for j, indirect_neighbor in enumerate(next_products[direct_neighbor]):
            if (
                product_prices[indirect_neighbor]
                > reservation_prices[indirect_neighbor]
            ):
                continue
            influence_graph[direct_neighbor, indirect_neighbor] = (
                lam**j * graph[direct_neighbor, indirect_neighbor]
            )

        influence_graph[primary_product, direct_neighbor] = (
            lam**i * graph[primary_product, direct_neighbor]
        )

    return influence_graph


def get_expected_influence_per_product(
    graph: NDArray[np.float64],
    product_prices: List[float],
    reservation_prices: List[float],
    next_products: List[Tuple[int, int]],
    lam: float,
) -> List[float]:
    """Get the expected influence of all products.

    This is a convenience method to call both `make_influence_graph` and
    `get_influence_of_seed` for all products, see those methods for more
    details.

    Returns:
        A list of product influence scores, essentially saying how many other
        products the product will lead to being bought on average (if it is
        bought).
    """
    product_influences = []
    for product in range(len(product_prices)):
        influence_graph = make_influence_graph(
            product,
            graph,
            product_prices,
            reservation_prices,
            next_products,
            lam,
        )
        product_influence, _ = get_influence_of_seed(influence_graph, product)
        product_influences.append(product_influence)

    return product_influences
