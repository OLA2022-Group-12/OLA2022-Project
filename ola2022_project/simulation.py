import tqdm
import random


def get_reward_from_interactions(interactions, prices):
    """Computes the margin made each day, for each of the 3 classes of users.

    Arguments:
        interactions: A list of tuples, with a size corresponding to num_customers. Every
        tuple denotes an interaction. tuple[0] is the user's class, which can be
        1, 2 or 3. tuple[1] is a numpy array of 5 elemets, where every element i
        represents how many of the products i+1 the customer bought

        prices: Price of the 5 products

    Returns:
        A list, with the size corresponding to the number of user classes (1,3)
        Return the sum of the margin made each day, for each of the 3 classes of users
    """

    reward_per_class = [0, 0, 0]

    for i in range(len(interactions)):

        # Get the class of the user
        user_class = interactions[i][0] - 1

        # Compute how much a customer purchased
        reward = sum([a * b for a, b in zip(interactions[i][1], prices)])

        reward_per_class[user_class] += reward

    return reward_per_class


def simulation(env, learner, prices, n_experiment=1, n_day=300):
    """Runs the simulation for a certain amount of experiments consisting of a
    certain amount of days

    Arguments:
        env:

        learner:

        prices: Prices of the 5 products. Shape (1,5)

        n_experiment: Number of times the experiment is performed,
          to have statistically more accurate results.
          By default, the value is 1 because in the real world we don't have
          time to do each experiment several times.

        n_day: Duration of the experience in days

    Returns:

    """

    reward_per_experiment = []

    for day in tqdm(range(n_day)):

        # Every day, there is a random number of potential new customers
        n_new_customers = random.randint(0, 100)

        budget = learner.budget

        # All the interactions of an entire day, depending on the budget
        interactions = env.get_day_of_interactions(n_new_customers, budget)

        reward = get_reward_from_interactions(interactions, prices)
        learner.update(budget, reward)

    reward_per_experiment.append(learner.collected_reward)
