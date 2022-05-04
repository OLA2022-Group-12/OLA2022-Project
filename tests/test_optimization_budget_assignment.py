import numpy as np
import pytest
from numpy.testing import assert_array_equal
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays


from ola2022_project.optimization import budget_assignment


def test_empty_input_gives_none():
    assert budget_assignment(np.array([[]])).size == 0


@pytest.mark.parametrize(
    "C,expected",
    [
        ([[0, 1, 2]], [2]),
        ([[0, 1, 2], [0, 3, 5]], [0, 2]),
        ([[0, 2, 3], [0, 2, 2]], [1, 1]),
        ([[0, 1, 2], [0, 3, 4], [0, 2, 4]], [0, 1, 1]),
        ([[0, 1, 2], [0, 3, 4], [0, 2, 5], [0, 4, 4]], [0, 1, 0, 1]),
        ([[0, 1, 2], [0, 3, 4], [0, 2, 5], [0, 4, 4], [0, 6, 7]], [0, 0, 0, 1, 1]),
        ([[0, 1, 2, 3], [0, 3, 5, 6]], [1, 2]),
        ([[0, 1, 2, 3, 4], [0, 3, 5, 6, 7]], [2, 2]),
        ([[0, 1, 2, 4, 4], [0, 3, 5, 5, 7], [1, 2, 3, 3, 3]], [3, 1, 0]),
    ],
)
def test_simple_examples(C, expected):
    assert_array_equal(np.array(expected), budget_assignment(np.array(C)))


@given(
    arrays(
        dtype=np.int32,
        elements=st.integers(0, 10000),
        shape=st.tuples(st.integers(1, 1), st.integers(1, 50)),
    ).map(lambda C: np.sort(C, axis=-1))
)
def test_single_campaign_always_choose_biggest_value(C):
    index_of_best = np.argmax(C, axis=-1)
    assert_array_equal(np.array(index_of_best), budget_assignment(C))


@given(
    arrays(
        dtype=np.int32,
        elements=st.integers(0, 10000),
        shape=st.tuples(st.integers(2, 2), st.integers(1, 50)),
    ).map(lambda C: np.sort(C, axis=-1))
)
def test_two_campaigns_biggest_value_where_budget_satisfied(C):
    _, m = np.shape(C)

    biggest_value = 0
    for i in range(m):
        for j in range(m):
            # Biggest value where the budget is not exceeded
            if i + j < m and C[0][i] + C[1][j] > biggest_value:
                biggest_value = C[0][i] + C[1][j]

    selected_allocs = budget_assignment(C)
    selected_value = (C * np.eye(m)[selected_allocs]).sum()

    assert biggest_value == selected_value


@given(
    arrays(
        dtype=np.int32,
        elements=st.integers(0, 10000),
        shape=st.tuples(st.integers(1, 10), st.integers(1, 50)),
    ).map(lambda C: np.sort(C, axis=-1))
)
def test_sum_of_allocs_should_not_exceed_budget(C):
    _, m = np.shape(C)
    selected_allocs = budget_assignment(C)

    assert m - 1 >= selected_allocs.sum()


@given(
    arrays(
        dtype=np.int32,
        elements=st.integers(0, 10000),
        shape=st.tuples(st.integers(1, 10), st.integers(1, 50)),
    ).map(lambda C: np.sort(C, axis=-1))
)
def test_value_is_between_biggest_single_value_and_sum_of_last(C):
    _, m = np.shape(C)

    selected_allocs = budget_assignment(C)
    selected_value = (C * np.eye(m)[selected_allocs]).sum()

    # Because we allocate at least the biggest value in a single campaign
    lower_bound = C.max()

    # Because we cannot possibly get a higher value than the sum of the last
    # columns of each campaign
    upper_bound = C[:, -1].sum()

    assert lower_bound <= selected_value
    assert upper_bound >= selected_value
