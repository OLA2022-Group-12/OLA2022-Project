import pytest
from hypothesis import given, strategies as st


from tests import generated_environment
from ola2022_project.environment.environment import (
    create_masked_environment,
    Step,
)
from ola2022_project.learners import ClairvoyantLearner


@given(
    step=st.sampled_from(Step).filter(lambda s: s != Step.ZERO),
    env=generated_environment(),
)
def test_with_masked_environment_variables(step, env):
    masked_env = create_masked_environment(step, env)
    learner = ClairvoyantLearner()

    with pytest.raises(RuntimeError, match="masked env"):
        learner.predict(masked_env)


@given(
    env=generated_environment(),
)
def test_sum_of_predictions_less_than_total_budget(env):
    masked_env = create_masked_environment(Step.ZERO, env)
    learner = ClairvoyantLearner()

    predictions = learner.predict(masked_env)

    assert predictions.sum() <= masked_env.total_budget
