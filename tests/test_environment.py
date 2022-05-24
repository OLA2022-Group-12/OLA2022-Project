import numpy as np
import pytest
from hypothesis import given, strategies as st


from ola2022_project.environment.environment import alpha_function


@given(
    budget=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0),
    steepness=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0),
    shift=st.floats(allow_nan=False, allow_infinity=False, max_value=100.0),
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
    shift=st.floats(allow_nan=False, allow_infinity=False, max_value=100.0),
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
