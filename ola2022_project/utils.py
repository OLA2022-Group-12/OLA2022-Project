import logging
from typing import List, Optional
import numpy as np


logger = logging.getLogger(__name__)


def calculate_aggregated_budget_value(
    product_graph_landing_values: List[float],
    product_prices: List[float],
    class_budget_alphas: np.ndarray,
    class_ratios: List[float],
    class_reservation_prices: Optional[List[float]] = None,
):
    """Calculates the class-aggregated budget value matrix

    Arguments:
        product_graph_landing_values: A list of the value of landing on a certain product
        product_prices: A list of the product prices
        class_budget_alphas: A (CxM) matrix of the estimated value alphas of a
        given class for the given budget steps (C is class and M is budget
        steps)
        class_ratios: A list of the class ratios
        class_reservation_prices: A optional list of the known reservation
        prices for class i + 1. If not known, will assume there is no limit

    Returns:
        An (PxM) matrix which has the aggregated budget values across the given
        classes, where P is the number of products and M is the number of budget
        steps.
    """
    n_products = len(product_prices)
    n_budget_steps = np.shape(class_budget_alphas)[1]

    # The aggregated budget value matrix which denotes the value of
    # assigning j (column) of the budget to product i (row)
    aggregated_budget_value = np.zeros((n_products, n_budget_steps))

    for user_class, (
        class_ratio,
        budget_alphas_for_class,
        class_reservation_price,
    ) in enumerate(
        zip(
            class_ratios,
            class_budget_alphas,
            class_reservation_prices or [None for _ in class_ratios],
        )
    ):
        logger.debug(
            f"Calculating budget value matrix for user_class '{user_class + 1}'..."
        )

        budget_value_for_class = np.array(
            [
                product_price * product_graph_landing_value * budget_alphas_for_class
                for product_price, product_graph_landing_value in zip(
                    product_prices, product_graph_landing_values
                )
            ]
        )

        # If the product price is greater than the reservation price, the
        # value of allocating more to this campaign is always zero (as this
        # user group will NEVER buy the product)
        if class_reservation_price is not None:
            for i, product_price in enumerate(product_prices):
                if product_price > class_reservation_price:
                    budget_value_for_class[i, :] = 0.0

        logger.debug(
            f"budget_value_for_class {budget_value_for_class.shape}: "
            + str(budget_value_for_class)
        )

        # Here we take the class ratio into account, so that if a class is
        # more present, it will have a larger impact on the final allocation
        aggregated_budget_value += class_ratio * budget_value_for_class

    logger.debug(
        f"aggregated_budget_value {aggregated_budget_value.shape}: "
        + str(aggregated_budget_value)
    )

    return aggregated_budget_value
