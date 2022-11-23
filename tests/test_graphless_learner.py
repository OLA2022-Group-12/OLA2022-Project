import pytest
from hypothesis import given, strategies as st
import numpy as np


from tests import generated_environment
from ola2022_project.environment.environment import (
    Interaction,
    create_masked_environment,
    Step,
)
from ola2022_project.learners import GraphlessLearner


@given(
    step=st.sampled_from([Step.ZERO, Step.ONE, Step.TWO]),
    env=generated_environment(),
)
def test_with_masked_environment_variables(step, env):
    masked_env = create_masked_environment(step, env)
    rng = np.random.default_rng()

    with pytest.raises(RuntimeError, match="masked env"):
        _ = GraphlessLearner(rng, 5, masked_env)


@given(
    env=generated_environment(),
)
def test_sum_of_predictions_less_than_total_budget(env):
    masked_env = create_masked_environment(Step.THREE, env)
    rng = np.random.default_rng()

    learner = GraphlessLearner(rng, 5, masked_env)

    predictions = learner.predict(masked_env)

    assert np.sum(predictions[0]) <= env.total_budget


@given(
    env=generated_environment(),
    primary_product=st.integers(min_value=0, max_value=4),
)
def test_present_edges_increase_alphas_and_absent_edges_decrease_betas(
    env, primary_product
):
    masked_env = create_masked_environment(Step.THREE, env)
    rng = np.random.default_rng()

    n_products = len(masked_env.product_prices)

    learner = GraphlessLearner(rng, 5, masked_env)

    items_bought = [1 if i == primary_product else 0 for i in range(n_products)]
    secondary_one = masked_env.next_products[primary_product][0]
    secondary_two = masked_env.next_products[primary_product][1]

    present_edges = [(primary_product, secondary_one)]
    absent_edges = [(primary_product, secondary_two)]

    interactions = [
        Interaction(
            user_features=[],
            user_class=0,
            items_bought=items_bought,
            landed_on=primary_product,
            edges=present_edges,
        )
    ]

    alpha_present_pre_learn = [learner.alpha_param[s, t] for s, t in present_edges]
    beta_absent_pre_learn = [learner.beta_param[s, t] for s, t in absent_edges]

    preds = learner.predict(masked_env)
    reward = 10  # TODO utils method to calculate this
    learner.learn(interactions, reward, preds)

    alpha_present_post_learn = [learner.alpha_param[s, t] for s, t in present_edges]
    beta_absent_post_learn = [learner.beta_param[s, t] for s, t in absent_edges]

    for alpha_pre_learn, alpha_post_learn in zip(
        alpha_present_pre_learn, alpha_present_post_learn
    ):
        assert alpha_post_learn > alpha_pre_learn, "present edge should increase alpha"

    for beta_pre_learn, beta_post_learn in zip(
        beta_absent_pre_learn, beta_absent_post_learn
    ):
        assert beta_post_learn > beta_pre_learn, "missing edge should decrease beta"
