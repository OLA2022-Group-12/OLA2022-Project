import numpy as np
from ola2022_project.environment.environment import EnvironmentData
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays


@st.composite
def generated_environment(draw):
    """Generates an environment with random data based on some assumtions"""

    max_budget_size = 1000
    min_classes = 3
    max_classes = 3
    min_products = 5
    max_products = 5
    max_items_size = 10
    max_product_price = 100

    total_budget_st = st.integers(min_value=1, max_value=max_budget_size)
    total_budget = draw(total_budget_st)

    classes_st = st.integers(min_value=min_classes, max_value=max_classes)
    n_classes = draw(classes_st)

    class_ratios_st = (
        st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=n_classes,
            max_size=n_classes,
        )
        # Perform softmax, which makes it into a probability distribution no
        # matter the integers chosen
        .map(np.array)
        .map(np.exp)
        .map(lambda x: x / np.sum(x))
        .map(lambda x: x.tolist())
    )
    class_ratios = draw(class_ratios_st)

    n_products_st = st.integers(min_value=min_products, max_value=max_products)
    n_products = draw(n_products_st)

    product_prices_st = st.lists(
        st.integers(min_value=1, max_value=max_product_price),
        min_size=n_products,
        max_size=n_products,
    )
    product_prices = draw(product_prices_st)
    max_product_price = np.max(np.array(product_prices))
    min_product_price = np.min(np.array(product_prices))

    classes_parameters_st = st.tuples(
        # Reservation price
        st.integers(min_value=min_product_price, max_value=max_product_price),
        # Steepness
        st.floats(min_value=0.1, max_value=0.9),
        # Shift
        st.floats(min_value=1.0, max_value=10.0),
        # Upper bound
        st.integers(min_value=10, max_value=100),
    )
    classes_parameters = draw(
        st.lists(classes_parameters_st, min_size=n_classes, max_size=n_classes)
    )

    competitor_budget_st = st.integers(min_value=0, max_value=max_budget_size)
    competitor_budget = draw(competitor_budget_st)

    lambda_st = st.floats(min_value=0.1, max_value=0.9)
    lam = draw(lambda_st)

    max_items_st = st.integers(min_value=1, max_value=max_items_size)
    max_items = draw(max_items_st)

    graph_st = arrays(
        dtype=np.float32,
        shape=(n_products, n_products),
        elements=st.floats(min_value=0.0, max_value=1.0, width=32),
    )
    graph_fully_connected_st = st.booleans()
    graph_mask_st = arrays(
        dtype=bool,
        shape=(n_products, n_products),
        # Because booleans will shrink towards False, we invert so that they
        # rather shrink towards True when the graph mask is shrinking.
        elements=st.booleans().map(lambda b: not b),
    )

    graph = draw(graph_st)
    if draw(graph_fully_connected_st):
        mask = draw(graph_mask_st)
        graph[mask] = 0.0

    # Remove autoloops
    np.fill_diagonal(graph, 0)

    next_products_st = st.lists(
        st.tuples(
            # Choose offset to the secondary products (hence 1 to n - 1)
            st.integers(min_value=1, max_value=n_products - 1),
            st.integers(min_value=1, max_value=n_products - 1),
            # Ensure secondary products are not equal
        ).filter(lambda x: x[0] != x[1]),
        min_size=n_products,
        max_size=n_products,
    ).map(
        # Ensures that a product never has itself as a secondary. This is
        # enabled by drawing between 1 and n_products - 1, which will then be
        # used as a offset for the secondary of the product product i.
        lambda ns: [
            ((i + x1) % n_products, (i + x2) % n_products)
            for i, (x1, x2) in enumerate(ns)
        ]
    )
    next_products = draw(next_products_st)

    return EnvironmentData(
        total_budget=total_budget,
        class_ratios=class_ratios,
        product_prices=product_prices,
        classes_parameters=classes_parameters,
        competitor_budget=competitor_budget,
        lam=lam,
        max_items=max_items,
        graph=graph,
        next_products=next_products,
    )
