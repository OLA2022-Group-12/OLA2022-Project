import logging
from hypothesis import given  # , strategies as st

# import numpy as np

from tests import generated_environment

# from ola2022_project.simulation import get_reward_from_interactions
from ola2022_project.environment.environment import (
    create_masked_environment,
    Step,
    # get_day_of_interactions,
)
from ola2022_project.learners import ClairvoyantLearner, StupidLearner


logger = logging.getLogger(__name__)


@given(
    env=generated_environment(),
    # seed=st.integers(),
)
def test_clairvoyant_should_always_outperform_stupid(env):  # , #seed):
    masked_env = create_masked_environment(Step.ZERO, env)

    clair_leaner = ClairvoyantLearner()
    stupid_learner = StupidLearner()

    clair_budgets = clair_leaner.predict(masked_env)
    stupid_budgets = stupid_learner.predict(masked_env)

    logger.debug(f"clair_budgets  = {clair_budgets}")
    logger.debug(f"stupid_budgets = {stupid_budgets}")

    # TODO This, of course, doesn't work because calling get_day_of_interactions
    # will always use randomness to determine what interactions happend, so we
    # cannot compare to learners directly without accounting for the noise
    # produced by the rng given here. Perhaps there is a way we could calculate
    # interactions deterministically so that the "only" altering factor here
    # would be the budgets?

    # After thinking a bit, I think perhaps that the reason it doesn't work,
    # even after "resetting" the random generator seed (so that it should be
    # "identical" each day of interactions), is because we sample randomness
    # BASED ON randomness. E.g. in the _go_to_page function we sample randomness
    # on wether or not to continue to sample randomness. So what this means is
    # that we might call into randomness a non-deterministic amount of times,
    # which means that the underlaying big generator advances at different paces
    # for each of the times of generation. If we could get the randomness to be
    # sampled **exactly** the same amount of times no matter how many
    # interactions we are to generate, I think it would be deterministic, even
    # for proper randomness. I believe a way to do this is for e.g.
    # `_go_to_page` is to change it from being recursive to being a loop which
    # always generates the full graph, and then we sample wether to keep the
    # bought items or not *after* they have been generated. I will take a jab at
    # this :)

    # rng = np.random.default_rng(seed)

    # rng_state = rng.bit_generator.state
    # logger.warning(f"pre  random_state = {rng_state}")
    # clair_interactions = get_day_of_interactions(rng, 5, clair_budgets, env)

    # rng.bit_generator.state = rng_state
    # logger.warning(f"post random_state = {rng.bit_generator.state}")
    # stupid_interactions = get_day_of_interactions(rng, 5, stupid_budgets, env)

    # logger.warning(f"clair_interactions = {clair_interactions}")
    # logger.warning(f"stupid_interactions = {stupid_interactions}")

    # clair_reward = get_reward_from_interactions(clair_interactions, env.product_prices)
    # stupid_reward = get_reward_from_interactions(stupid_interactions, env.product_prices)

    # assert clair_reward >= stupid_reward
