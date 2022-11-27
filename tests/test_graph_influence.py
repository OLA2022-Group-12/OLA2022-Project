import numpy as np
from hypothesis import given, strategies as st

from ola2022_project.environment.environment import EnvironmentData
from ola2022_project.optimization import get_expected_value_per_node

from ola2022_project.optimization.graph_influence import get_influence_of_seed

from tests import generated_environment


# @given(
#    env=generated_environment(),
# )
# def test_expected_value_per_node_returns_boundend_output(env: EnvironmentData):
#    user_class = 0
#    result = get_expected_value_per_node(
#        env.graph,
#        env.product_prices,
#        [p.reservation_price for p in env.classes_parameters[user_class]],
#        env.next_products,
#        env.lam,
#    )
#
#    assert len(result) == len(env.product_prices), "equal as amount of products"
#
#    for item in result:
#        assert 0.0 <= item <= sum(env.product_prices), "within maximum reward"


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_single_direct_neighbor(factor):
    graph = np.zeros((2, 2))

    graph[0, 1] = factor

    influence = get_influence_of_seed(graph, 0)

    np.testing.assert_approx_equal(factor, influence)


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_single_secondary_neighbors(factor):
    graph = np.zeros((3, 3))

    # Direct
    graph[0, 1] = factor
    # Secondary
    graph[1, 2] = factor

    influence = get_influence_of_seed(graph, 0)

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

    influence = get_influence_of_seed(graph, 0)

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

    influence = get_influence_of_seed(graph, 0)

    np.testing.assert_approx_equal(factor + factor**2 + 2 * factor**3, influence)


@given(
    st.integers(min_value=2, max_value=10),
)
def test_self_loops_give_no_influence(n):
    graph = np.identity(n)
    influence = get_influence_of_seed(graph, 0)
    np.testing.assert_approx_equal(0.0, influence)


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_single_direct_neighbor_circular(factor):
    graph = np.zeros((2, 2))

    graph[0, 1] = factor
    graph[1, 0] = factor

    influence = get_influence_of_seed(graph, 0)

    np.testing.assert_approx_equal(factor, influence)


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_two_direct_neighbor_circular(factor):
    graph = np.zeros((3, 3))

    # Direct
    graph[0, 1] = factor
    graph[0, 2] = factor

    # Indirect (but shouldn't affect probability)
    graph[1, 2] = factor

    influence = get_influence_of_seed(graph, 0)

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

    influence = get_influence_of_seed(graph, 0)
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

    influence = get_influence_of_seed(
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
