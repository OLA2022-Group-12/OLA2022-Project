import numpy as np
import pytest
from hypothesis import given, strategies as st


from ola2022_project.environment.environment import alpha_function


@given(
    budget=st.floats(
        allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1e6
    ),
    max_useful_budget=st.floats(
        allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1e6
    ),
    upper_bound=st.floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=0.0,
        max_value=1.0,
        exclude_min=True,
    ),
)
@pytest.mark.filterwarnings("ignore:overflow encountered")
def test_alpha_function_always_between_zero_and_upper_bound(
    budget, max_useful_budget, upper_bound
):
    # No matter the budget fraction between 0 and 1, the assigned budget/alpha
    # value should always be between zero and the upper bound (any non-zero value)
    assert 0.0 <= alpha_function(budget, upper_bound, max_useful_budget) <= upper_bound


@given(
    x_n=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0),
    x_m=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0),
    max_useful_budget=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0),
    upper_bound=st.floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=0.0,
        max_value=1.0,
        exclude_min=True,
    ),
)
@pytest.mark.filterwarnings("ignore:overflow encountered")
def test_alpha_function_is_increasing(x_n, x_m, max_useful_budget, upper_bound):
    # The alpha function should always be increasing, hence if a value of
    # budget (x_0) is smaller than another (x_1), it should produce a
    # lower or equal value the alpha function is called with the other
    # parameters fixed
    x_0 = np.minimum(x_n, x_m)
    x_1 = np.maximum(x_n, x_m)

    assert alpha_function(x_0, upper_bound, max_useful_budget) <= alpha_function(
        x_1, upper_bound, max_useful_budget
    )
