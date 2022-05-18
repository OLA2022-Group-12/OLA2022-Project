import tqdm
from ola2022_project.environment.environment import get_day_of_interactions


def get_reward_from_interactions(interactions, prices):
    """Computes the margin made each day, for each of the 3 classes of users.

    Arguments:
        interactions: A list of tuples, with a size corresponding to num_customers. Every
        tuple denotes an interaction. tuple[0] is the user's class, which can be
        1, 2 or 3. tuple[1] is a numpy array of 5 elemets, where every element i
        represents how many of the products i+1 the customer bought

        prices: Price of the 5 products

    Returns:
        The list, with the size corresponding to the number of user classes
        + 1, return the sum of the margin made each day, for each of the 3
        classes of users (and the zeroth place is the user-classes entries,
        aggregated).
    """

    reward_per_class = [0, 0, 0, 0]

    for interaction in enumerate(interactions):

        # Compute how much a customer purchased
        reward = sum([a * b for a, b in zip(interaction[1], prices)])

        # Get the user class and add the rewards. The zeroth place will be the
        # rewards without a specific user class
        user_class = interaction[0]
        reward_per_class[user_class] += reward

    return reward_per_class


def simulation(rng, env, learner_factory, prices, n_experiment=1, n_day=300):
    """Runs the simulation for a certain amount of experiments consisting of a
    certain amount of days

    Arguments:
        env:

        learner_factory: A function which creates a new learner (needed to run
        multiple experiments with "fresh" learners)

        prices: Prices of the 5 products. Shape (1,5)

        n_experiment: Number of times the experiment is performed,
          to have statistically more accurate results.
          By default, the value is 1 because in the real world we don't have
          time to do each experiment several times.

        n_day: Duration of the experience in days

    Returns:
        The collected rewards of running each experiment

    """

    rewards_per_experiment = []

    for _ in tqdm.trange(n_experiment, desc="experiment"):
        # Create a new learner for each experiment
        learner = learner_factory()

        for _ in tqdm.trange(n_day, desc="day"):
            # Every day, there is a random number of potential new customers
            n_new_customers = rng.integers(0, 100)

            budgets = learner.budgets

            # All the interactions of an entire day, depending on the budget
            interactions = get_day_of_interactions(rng, n_new_customers, budgets, env)

            rewards = get_reward_from_interactions(interactions, prices)
            learner.update(budgets, rewards)

        rewards_per_experiment.append(learner.collected_reward)

    return rewards_per_experiment
