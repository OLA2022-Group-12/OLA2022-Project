import numpy as np
import pytest
from numpy.testing import assert_array_equal
from hypothesis import given, strategies as st, register_random
from hypothesis.extra.numpy import arrays


from ola2022_project.environment.environment import alpha_function, generate_graph


@given(
    budget_fraction=st.floats(
        allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1.0
    ),
    steepness=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0),
    shift=st.floats(allow_nan=False, allow_infinity=False),
    total_budget=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0),
)
@pytest.mark.filterwarnings("ignore:overflow encountered")
def test_alpha_function_always_between_zero_and_total_budget(
    budget_fraction, steepness, shift, total_budget
):
    # No matter the budget fraction between 0 and 1, the assigned budget/alpha
    # value should always be between zero and the total budget (any non-zero value)
    assert (
        0.0
        <= alpha_function(budget_fraction, steepness, shift, total_budget)
        <= total_budget
    )


@given(
    x_n=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1.0),
    x_m=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1.0),
    steepness=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0),
    shift=st.floats(allow_nan=False, allow_infinity=False),
    total_budget=st.floats(allow_nan=False, allow_infinity=False, min_value=0.0),
)
@pytest.mark.filterwarnings("ignore:overflow encountered")
def test_alpha_function_is_increasing(x_n, x_m, steepness, shift, total_budget):
    # The alpha function should always be increasing, hence if a value of
    # budget_fraction (x_0) is smaller than another (x_1), it should produce a
    # lower or equal value the alpha function is called with the other
    # parameters fixed
    x_0 = np.minimum(x_n, x_m)
    x_1 = np.maximum(x_n, x_m)

    assert alpha_function(x_0, steepness, shift, total_budget) <= alpha_function(
        x_1, steepness, shift, total_budget
    )
