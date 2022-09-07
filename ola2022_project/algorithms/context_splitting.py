import itertools
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from collections import namedtuple
from ola2022_project.utils import compute_hoeffding_bound
from ola2022_project.simulation import (
    dataset_simulation,
    DatasetSimParameters,
)


# TODO: remove
# Temporary named tuple utilized to represent a feature in the context generation
UserFeature = namedtuple("UserFeature", ["feature", "value"])


@dataclass
class Context:

    """Dataclass containing all the information related to contexts used
    in contextual generation. Its main purpose is to facilitate feature
    splitting algorithms by providing an interface capable of enclosing
    all the relevant information for choosing and evaluating different
    scenarios.
    """

    # List of current context features
    features: List[UserFeature]

    # Number of samples found in the training dataset for each feature that is
    # relevant to the context
    nums: List[int]

    # Expected appearance probabilities for the feature values identified by the context
    exp_prob: float

    # Maximum expected reward
    max_exp_reward: float

    # Probability-weighted lower Hoeffding bound for the maximum expected reward
    weighted_bound: float


def compute_weighted_bound(p: float, mu: float) -> float:

    """Computes the probability-weighted lower Hoeffding bound for the expected reward
    of a given probability-expected_reward pair.

    Arguments:
        p: probability

        mu: expected_reward

    Returns:
        Floating value corresponding to the weighted lower bound
    """

    return (p - compute_hoeffding_bound(p)) * (mu - compute_hoeffding_bound(mu))


def split_condition(
    context_1: Context, context_2: Context, base_context: Context
) -> float:

    """Computes the marginal difference for the split between contexts w.r.t. the base_context,
    often utilized to determined if a given feature split is worth applying.

    Arguments:
        context_1: first split context to evaluate

        context_2: second (complementary of the first) split context to evaluate

        base_context: base unsplitted context as a reference needed to calculate the
            marginal difference

    Returns:
        A floating value that contains the resulting value and can be evaluated as
        'split_condition >= 0' in order to obtain the corresponding truth value
    """

    return (
        context_1.weighted_bound
        + context_2.weighted_bound
        - base_context.weighted_bound
    )


def feature_filter(dataset, features):

    """Filters the elements of a dataset given a set of wanted features.

    Arguments:
        dataset: dataset to filter

        features: features to discriminate

    Returns:
        A new dataset composed only of interactions that satisfy the given features
    """

    filtered_dataset = []
    for dataset_day in dataset:
        filtered_dataset.append(
            list(
                filter(
                    lambda interaction: all(
                        lambda feature: feature in interaction.user_features
                    )
                ),
                dataset_day,
            )
        )
    return filtered_dataset


def feature_split(
    sim_param: DatasetSimParameters, dataset, context: Context, feature=None
) -> List[Context]:

    """Generates, trains and evaluates over a given dataset new contexts based on a feature split,
    starting from a base context and resulting in a binary split, therefore it creates two new
    contexts with a dependency on the given feature. If the context and the feature are not
    present, the function generates a single new base aggregated context over the dataset.

    Arguments:
        sim_param: parameters used by the dataset simulation to operate the learner and the
            environment

        dataset: current offline dataset gathered over a span of time containing samples
            used to train new models and define context attributes

        context: context utilized as a starting base for the feature split

        feature: feature that is being split upon

    Returns:
        A list of resulting contexts that may contain one to two elements based on the
        given arguments
    """

    if not context or not feature:
        pass
        # TODO base case

    feature_1 = UserFeature(feature.feature, 0)
    feature_2 = UserFeature(feature.feature, 1)

    return [
        feature_half_split(sim_param, dataset, context, feature_1),
        feature_half_split(sim_param, dataset, context, feature_2),
    ]


def feature_half_split(
    sim_param, dataset, context: Context, feature: UserFeature
) -> Context:

    """Generates, trains and evaluates over a given dataset a new context based on half of a feature
    binary split.

    Arguments:
        sim_param: parameters used by the dataset simulation to operate the learner and the
            environment

        dataset: current offline dataset gathered over a span of time containing samples
            used to train new models and define context attributes

        context: context utilized as a starting base for the feature split

        feature: feature that is being split upon

    Returns:
        A new context trained and evaluated over the filtered dataset
    """

    dataset_split = feature_filter(dataset, feature)
    n = np.sum(context.nums)
    n_split = [len(d) for d in dataset_split]
    features = np.concatenate(context.features, feature)

    # reward = dataset_simulation(sim_param, dataset)
    dataset_simulation(sim_param, dataset)
    exp_prob = n / np.sum(n_split)
    max_exp_reward = 0  # TODO: max expected reward

    return Context(
        features,
        n_split,
        exp_prob,
        max_exp_reward,
        compute_weighted_bound(exp_prob, max_exp_reward),
    )


def tree_generation(
    sim_param: DatasetSimParameters, dataset, features: List[UserFeature]
) -> List[Context]:

    """Computes the feature tree for a given dataset and set of features, it utilizes a
    greedy approach over feature splits by iteratively splitting on the most promising
    feature for each tree level; the splitting condition takes into consideration the
    lower bounds for the deciding factors since a pessimistic approach is needed in order
    to control the costly operation of splitting a context into sub-contexts. Being a greedy
    approach, the result holds no guarantee of optimality but underlines the most important
    features of the dataset effectively.

    Arguments:
        sim_param: parameters used by the dataset simulation to operate the learner and the
            environment

        dataset: current offline dataset gathered over a span of time, containing samples
            used to train new models and define context attributes

        features: list of all the possible features of the samples contained in the dataset

    Returns:
        A list containing the contexts chosen by the algorithm
    """

    # Get the base context data
    base_context = feature_split(sim_param, dataset)
    # Start generating tree recursively
    optimal_contexts = generate_tree_node(sim_param, dataset, features, base_context)
    return optimal_contexts if optimal_contexts else base_context


def generate_tree_node(
    sim_param: DatasetSimParameters,
    dataset,
    unsplit_features: List[UserFeature],
    base_context: Context,
) -> Optional[List[Context]]:

    """Recursive step for the feature tree generation; generates a new bifurcation inside
    the tree structure by choosing a feature to split the base context upon and recursively
    calling itself two times (binary split) using the resulting contexts as a base for each
    new node. The stopping criterion is reached when a node cannot split on any feature
    without creating a set of contexts that in the worst case perform worse than their related
    base problem (metric evaluated by applying a splitting condition). The resulting "optimal"
    contexts travel up the recursion chain up to the base of the tree, where the context structure
    containing the set of "optimal" contexts is returned.

    Arguments:
        sim_param: parameters used by the dataset simulation to operate the learner and the
            environment

        dataset: current offline dataset gathered over a span of time, containing samples
            used to train new models and define context attributes

        unsplit_features: list of features that have yet to be split upon

        base_context: reference base context for a split

    Returns:
        A list of "greedy optimal" contexts for the subproblem
    """

    # Generate all the possible splits for a single feature
    split_contexts = list(
        map(
            lambda feature: feature_split(sim_param, dataset, base_context, feature),
            unsplit_features,
        )
    )
    # Map the split contexts over a tuple containing the marginal difference of the split
    # condition and the related contexts
    split_contexts = list(
        map(
            lambda contexts: (
                contexts,
                split_condition(contexts, base_context),
            ),
            split_contexts,
        )
    )
    # Filter the split contexts based on the split_condition in order to only evaluate the
    # ones that are worth splitting
    # The split_condition is on the right side of the tuple, hence, it is accessed by using [1]
    split_contexts = list(
        filter(lambda t_contexts_split: t_contexts_split[1] >= 0), split_contexts
    )

    if split_contexts:
        # Take the split with best marginal value
        best_split, _ = max(
            split_contexts, key=lambda t_contexts_split: t_contexts_split[1]
        )
        # Compute the new set of unsplit features by adding the last feature added to one of the
        # contexts (which one is indifferent, here we chose the first one: index 0)
        new_features = list(unsplit_features)
        new_features.remove(best_split[0].features[-1])

        # Helper function to map recursive call results
        def none_context(context, result):
            return result if result else context

        # Call recursively the tree function and create tuples
        ret_contexts = list(
            map(
                lambda context: none_context(
                    generate_tree_node(sim_param, dataset, new_features, context),
                    context,
                ),
                best_split,
            )
        )
        # Return flattened list
        return list(itertools.chain(*ret_contexts))
    return None
