import numpy as np
from hypothesis import given, strategies as st

from ola2022_project.optimization.graph_influence import (
    get_influence_of_seed,
    make_influence_graph,
)


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_single_direct_neighbor(factor):
    graph = np.zeros((2, 2))

    graph[0, 1] = factor

    influence, _ = get_influence_of_seed(graph, 0)

    np.testing.assert_approx_equal(factor, influence)


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_single_secondary_neighbors(factor):
    graph = np.zeros((3, 3))

    # Direct
    graph[0, 1] = factor
    # Secondary
    graph[1, 2] = factor

    influence, _ = get_influence_of_seed(graph, 0)

    np.testing.assert_approx_equal(factor + factor**2, influence)


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_single_tertiary_neighbors(factor):
    graph = np.zeros((4, 4))

    # Direct
    graph[0, 1] = factor

    # Secondary
    graph[1, 2] = factor

    # Tertiary
    graph[2, 3] = factor

    influence, _ = get_influence_of_seed(graph, 0)

    np.testing.assert_approx_equal(factor + factor**2 + factor**3, influence)


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_leaf_tertiary_neighbors(factor):
    graph = np.zeros((5, 5))

    # Direct
    graph[0, 1] = factor

    # Secondary
    graph[1, 2] = factor

    # Tertiary
    graph[2, 3] = factor
    graph[2, 4] = factor

    influence, _ = get_influence_of_seed(graph, 0)

    np.testing.assert_approx_equal(factor + factor**2 + 2 * factor**3, influence)


@given(
    st.integers(min_value=2, max_value=10),
)
def test_self_loops_give_no_influence(n):
    graph = np.identity(n)
    influence, _ = get_influence_of_seed(graph, 0)
    np.testing.assert_approx_equal(0.0, influence)


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_single_direct_neighbor_circular(factor):
    graph = np.zeros((2, 2))

    graph[0, 1] = factor
    graph[1, 0] = factor

    influence, _ = get_influence_of_seed(graph, 0)

    np.testing.assert_approx_equal(factor, influence)


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_two_direct_neighbor_circular(factor):
    graph = np.zeros((3, 3))

    # Direct
    graph[0, 1] = factor
    graph[0, 2] = factor

    # Indirect (but shouldn't affect probability)
    graph[1, 2] = factor

    influence, _ = get_influence_of_seed(graph, 0)

    np.testing.assert_approx_equal(2 * factor, influence)


@given(
    st.integers(min_value=2, max_value=10),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_arbitary_size_graph(n, factor):
    graph = np.full((n, n), fill_value=factor)

    n_direct_neighbors = n - 1
    n_second_neighbors = max(0, (n - n_direct_neighbors - 1))
    n_third_neighbors = max(0, (n - n_second_neighbors - n_direct_neighbors - 1))

    influence, _ = get_influence_of_seed(graph, 0)
    np.testing.assert_approx_equal(
        n_direct_neighbors * factor
        + n_second_neighbors * factor**2
        + n_third_neighbors * factor**3,
        influence,
    )


@given(
    st.integers(min_value=2, max_value=10),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_cached_square_and_cubed_graphs(n, factor):
    graph = np.full((n, n), fill_value=factor)

    n_direct_neighbors = n - 1
    n_second_neighbors = max(0, (n - n_direct_neighbors - 1))
    n_third_neighbors = max(0, (n - n_second_neighbors - n_direct_neighbors - 1))

    graph_squared = np.linalg.matrix_power(graph, 2)
    graph_cubed = np.linalg.matrix_power(graph, 3)

    influence, _ = get_influence_of_seed(
        graph,
        0,
        graph_squared=graph_squared,
        graph_cubed=graph_cubed,
    )
    np.testing.assert_approx_equal(
        n_direct_neighbors * factor
        + n_second_neighbors * factor**2
        + n_third_neighbors * factor**3,
        influence,
    )


def test_low_reservation_price_gives_no_influence():
    env_graph = np.array([[0, 0.5, 0.2], [0.5, 0, 0.2], [0.5, 0.2, 0]])
    env_product_prices = [10.0, 20.0, 30.0]
    env_reservation_prices = [5.0, 5.0, 5.0]
    env_next_products = [(1, 2), (0, 2), (0, 1)]
    env_lam = 0.5

    for product in range(len(env_product_prices)):
        influence_graph = make_influence_graph(
            product,
            env_graph,
            env_product_prices,
            env_reservation_prices,
            env_next_products,
            env_lam,
        )
        np.testing.assert_almost_equal(0.0, influence_graph)


def test_product_influence_when_buying_all_products():
    env_graph = np.array([[0, 0.5, 0.2], [0.5, 0, 0.2], [0.5, 0.2, 0]])
    env_product_prices = [10.0, 20.0, 30.0]
    env_reservation_prices = [40.0, 20.0, 30.0]
    env_next_products = [(1, 2), (0, 2), (0, 1)]
    env_lam = 0.5

    for product in range(len(env_product_prices)):
        influence_graph = make_influence_graph(
            product,
            env_graph,
            env_product_prices,
            env_reservation_prices,
            env_next_products,
            env_lam,
        )
        product_influence, _ = get_influence_of_seed(influence_graph, product)
        expected = (
            env_graph[product, env_next_products[product][0]]
            + env_lam * env_graph[product, env_next_products[product][1]]
        )
        assert np.isclose(expected, product_influence)
