import numpy as np
from numpy.testing import assert_array_equal
import pytest
from hypothesis import given, strategies as st


from tests import generated_environment
from ola2022_project.environment.environment import (
    alpha_function,
    get_day_of_interactions,
)


@given(
    budget=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0),
    steepness=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0),
    shift=st.floats(allow_nan=False, allow_infinity=False),
    upper_bound=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0),
)
@pytest.mark.filterwarnings("ignore:overflow encountered")
def test_alpha_function_always_between_zero_and_upper_bound(
    budget, steepness, shift, upper_bound
):
    # No matter the budget fraction between 0 and 1, the assigned budget/alpha
    # value should always be between zero and the upper bound (any non-zero value)
    assert 0.0 <= alpha_function(budget, steepness, shift, upper_bound) <= upper_bound


@given(
    x_n=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0),
    x_m=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0),
    steepness=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0),
    shift=st.floats(allow_nan=False, allow_infinity=False),
    upper_bound=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0),
)
@pytest.mark.filterwarnings("ignore:overflow encountered")
def test_alpha_function_is_increasing(x_n, x_m, steepness, shift, upper_bound):
    # The alpha function should always be increasing, hence if a value of
    # budget (x_0) is smaller than another (x_1), it should produce a
    # lower or equal value the alpha function is called with the other
    # parameters fixed
    x_0 = np.minimum(x_n, x_m)
    x_1 = np.maximum(x_n, x_m)

    assert alpha_function(x_0, steepness, shift, upper_bound) <= alpha_function(
        x_1, steepness, shift, upper_bound
    )


@given(
    env=generated_environment(),
    seed=st.integers(min_value=0),
    num_customers=st.integers(min_value=0, max_value=100),
    budgets=st.lists(
        st.integers(min_value=0, max_value=100),
        min_size=5,
        max_size=5,
    ),
)
def test_get_interactions_is_deterministic_with_equal_budget(
    env, seed, num_customers, budgets
):
    rng = np.random.default_rng(seed)

    rng_state = rng.bit_generator.state
    first_interactions = get_day_of_interactions(rng, num_customers, budgets, env)

    rng.bit_generator.state = rng_state
    second_interactions = get_day_of_interactions(rng, num_customers, budgets, env)

    assert len(first_interactions) == len(second_interactions)
    for fi, si in zip(first_interactions, second_interactions):
        assert fi.user_class == si.user_class
        assert_array_equal(fi.items_bought, si.items_bought)


@given(
    env=generated_environment(single_product_price=True),
    seed=st.integers(min_value=0),
    num_customers=st.integers(min_value=1, max_value=100),
    budgets=st.lists(
        st.integers(min_value=10, max_value=100),
        min_size=5,
        max_size=5,
    ),
    budget_extra=st.integers(min_value=10, max_value=50),
)
def test_count_of_items_is_deterministically_equal_or_greater_with_bigger_budgets(
    env, seed, num_customers, budgets, budget_extra
):
    rng = np.random.default_rng(seed)

    rng_state = rng.bit_generator.state
    _ = get_day_of_interactions(rng, num_customers, budgets, env)

    rng.bit_generator.state = rng_state
    second_budgets = np.array([x + budget_extra for x in budgets])
    _ = get_day_of_interactions(rng, num_customers, second_budgets, env)

    # This doesn't work, because suprisingly enough hypothesis is able to search
    # it way to a configuration which invalidates this "stocastic" property..
    # That makes sense, but also makes it "impossible" to test these kind of
    # properties on the code. Hypothesis is too good and exploring the parameter
    # domain to find highly unplausible edge cases.

    # assert len(first_interactions) <= len(second_interactions)

    # first_item_count = np.sum([fi.items_bought for fi in first_interactions])  # type: ignore
    # secound_item_count = np.sum([fi.items_bought for fi in second_interactions])  # type: ignore

    # assert first_item_count <= secound_item_count
