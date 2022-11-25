import numpy as np
import pytest
from numpy.testing import assert_array_equal
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays


from ola2022_project.optimization import budget_assignment


def test_empty_input_gives_none():
    assert budget_assignment(np.array([[]])).size == 0


# fmt: off
@pytest.mark.parametrize(
    "c,expected",
    [
        ([[0., 1., 2.]], [2]),
        ([[0., 1., 2.], [0., 3., 5.]], [0, 2]),
        ([[0., 2., 3.], [0., 2., 2.]], [1, 1]),
        ([[0., 1., 2.], [0., 3., 4.], [0., 2., 4.]], [0, 1, 1]),
        ([[0., 1., 2.], [0., 3., 4.], [0., 2., 5.], [0., 4., 4.]], [0, 1, 0, 1]),
        ([[0., 1., 2.], [0., 3., 4.], [0., 2., 5.], [0., 4., 4.], [0., 6., 7.]], [0, 0, 0, 1, 1]),
        ([[0., 1., 2., 3.], [0., 3., 5., 6.]], [1, 2]),
        ([[0., 1., 2., 3., 4.], [0., 3., 5., 6., 7.]], [2, 2]),
        ([[0., 1., 2., 4., 4.], [0., 3., 5., 5., 7.], [1., 2., 3., 3., 3.]], [3, 1, 0]),
    ],
)
# fmt: on
def test_simple_examples(c, expected):
    assert_array_equal(np.array(expected), budget_assignment(np.array(c)))


@given(
    arrays(
        dtype=np.float64,
        elements=st.floats(
            min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False
        ),
        shape=st.tuples(st.integers(1, 1), st.integers(1, 50)),
    ).map(lambda c: np.sort(c, axis=-1))
)
def test_single_campaign_always_choose_biggest_value(c):
    index_of_best = np.argmax(c, axis=-1)
    assert_array_equal(np.array(index_of_best), budget_assignment(c))


@given(
    arrays(
        dtype=np.float64,
        elements=st.floats(
            min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False
        ),
        shape=st.tuples(st.integers(2, 2), st.integers(1, 50)),
    ).map(lambda c: np.sort(c, axis=-1))
)
def test_two_campaigns_biggest_value_where_budget_satisfied(c):
    _, m = np.shape(c)

    biggest_value = 0
    for i in range(m):
        for j in range(m):
            # Biggest value where the budget is not exceeded
            if i + j < m and c[0][i] + c[1][j] > biggest_value:
                biggest_value = c[0][i] + c[1][j]

    selected_allocs = budget_assignment(c)
    selected_value = (c * np.eye(m)[selected_allocs]).sum()

    assert biggest_value == selected_value


@given(
    arrays(
        dtype=np.float64,
        elements=st.floats(
            min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False
        ),
        shape=st.tuples(st.integers(1, 10), st.integers(1, 50)),
    ).map(lambda c: np.sort(c, axis=-1))
)
def test_sum_of_allocs_should_not_exceed_budget(c):
    _, m = np.shape(c)
    selected_allocs = budget_assignment(c)

    assert m - 1 >= selected_allocs.sum()


@given(
    arrays(
        dtype=np.float64,
        elements=st.floats(
            min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False
        ),
        shape=st.tuples(st.integers(1, 10), st.integers(1, 50)),
    ).map(lambda c: np.sort(c, axis=-1))
)
def test_value_is_between_biggest_single_value_and_sum_of_last(c):
    _, m = np.shape(c)

    selected_allocs = budget_assignment(c)
    selected_value = (c * np.eye(m)[selected_allocs]).sum()

    # Because we allocate at least the biggest value in a single campaign
    lower_bound = c.max()

    # Because we cannot possibly get a higher value than the sum of the last
    # columns of each campaign
    upper_bound = c[:, -1].sum()

    assert lower_bound < selected_value or np.isclose(lower_bound, selected_value)
    assert upper_bound > selected_value or np.isclose(upper_bound, selected_value)


@given(
    arrays(
        dtype=np.float64,
        elements=st.floats(
            min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False
        ),
        shape=st.tuples(st.integers(1, 10), st.integers(10, 1000)),
    ).map(lambda c: np.sort(c, axis=-1)),
    st.integers(100, 1000),
)
def test_always_uses_less_than_budget(c, total_budget):
    _, m = np.shape(c)

    selected_allocs = budget_assignment(c)

    budget_steps = np.linspace(0, total_budget, m)

    used_budget = budget_steps[selected_allocs].sum()

    assert used_budget < total_budget or np.isclose(total_budget, used_budget)


# TODO change to test single examples
# def test_a():
#     c = np.array(
#         [
#             [
#                 0.0,
#                 4.86829846,
#                 7.87383635,
#                 9.74581001,
#                 10.92067002,
#                 11.66281403,
#                 12.13417922,
#                 12.43492369,
#                 12.627528,
#                 12.75125702,
#                 12.83094098,
#                 12.8823646,
#                 12.91560619,
#                 12.93712379,
#                 12.95106789,
#                 12.96011237,
#                 12.96598325,
#                 12.96979648,
#                 12.97227453,
#                 12.9738856,
#                 12.97493341,
#             ],
#             [
#                 0.0,
#                 1.62252438,
#                 2.59892388,
#                 3.1981744,
#                 3.57276055,
#                 3.8108004,
#                 3.96425112,
#                 4.0643768,
#                 4.13036372,
#                 4.17420423,
#                 4.20351862,
#                 4.223219,
#                 4.23651037,
#                 4.24550484,
#                 4.25160563,
#                 4.25575099,
#                 4.25857147,
#                 4.26049245,
#                 4.2618018,
#                 4.26269478,
#                 4.26330407,
#             ],
#             [
#                 0.0,
#                 2.96615653,
#                 5.15122537,
#                 6.76264053,
#                 7.95230728,
#                 8.83157684,
#                 9.48215552,
#                 9.96405881,
#                 10.32141584,
#                 10.58670907,
#                 10.7838739,
#                 10.93056679,
#                 11.03982672,
#                 11.12129339,
#                 11.18210133,
#                 11.22753679,
#                 11.26152091,
#                 11.2869655,
#                 11.30603519,
#                 11.32034095,
#                 11.331083,
#             ],
#             [
#                 0.0,
#                 0.77090547,
#                 1.2164954,
#                 1.49856206,
#                 1.68869623,
#                 1.82192898,
#                 1.91742366,
#                 1.98677087,
#                 2.03752738,
#                 2.07486656,
#                 2.10243532,
#                 2.12284867,
#                 2.13800073,
#                 2.14927216,
#                 2.15767382,
#                 2.16394826,
#                 2.16864246,
#                 2.17216035,
#                 2.17480093,
#                 2.17678597,
#                 2.17828035,
#             ],
#             [
#                 0.0,
#                 0.20498198,
#                 0.34260975,
#                 0.43637874,
#                 0.50129964,
#                 0.54702032,
#                 0.57978778,
#                 0.60368331,
#                 0.62140181,
#                 0.63474493,
#                 0.64493423,
#                 0.6528109,
#                 0.65896402,
#                 0.66381324,
#                 0.66766283,
#                 0.67073706,
#                 0.67320396,
#                 0.67519116,
#                 0.6767969,
#                 0.67809762,
#                 0.67915334,
#             ],
#         ]
#     )
#
#     expected_allocs = []
#
#     _, m = np.shape(c)
#
#     n_budget_steps = 20
#     budget_steps = np.linspace(0, 400, n_budget_steps + 1)
#
#     expected_allocs = np.array([11, 0, 8, 0, 0])
#     selected_allocs = budget_assignment(c)
#
#     reward_values = (c * np.eye(m)[selected_allocs]).sum(axis=-1)
#     expected_reward_values = (c * np.eye(m)[expected_allocs]).sum(axis=-1)
#
#     total_reward = reward_values.sum()
#     expected_total_reward = expected_reward_values.sum()
#
#     print("Selected")
#
#     print(selected_allocs)
#     print(reward_values)
#     print(total_reward)
#     print(budget_steps[selected_allocs].sum())
#
#     print()
#     print("Expected")
#
#     print(expected_allocs)
#     print(expected_reward_values)
#     print(expected_total_reward)
#     print(budget_steps[expected_allocs].sum())
#
#     assert False
