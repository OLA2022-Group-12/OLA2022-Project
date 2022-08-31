from hypothesis import given

from ola2022_project.environment.environment import EnvironmentData
from ola2022_project.optimization import get_expected_value_per_node

from tests import generated_environment


@given(
    env=generated_environment(),
)
def test_expected_value_per_node_returns_boundend_output(env: EnvironmentData):
    result = get_expected_value_per_node(0, env)

    assert len(result) == len(env.product_prices), "equal as amount of products"

    for item in result:
        assert 0.0 <= item <= sum(env.product_prices), "within maximum reward"
