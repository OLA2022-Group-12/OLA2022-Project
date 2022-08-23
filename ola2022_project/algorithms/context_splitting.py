import math
import itertools
from dataclasses import dataclass
from typing import List


@dataclass
class Context:
    features: List[int]  # List of current context features
    last_feature: int  # Last added feature
    exp_probs: List[float]  # Expected probabilities of finding a given feature
    nums: List[int]  # Number of samples found with a given feature
    max_exp_reward: float  # Maximum expected reward
    weighted_bound: float  # Weighted lower Hoeffding bound for the maximum expected reward


def compute_hoeffding_bound(n, confidence=0.05):
    return math.sqrt(-math.log(confidence) / (2 * n))


def compute_weighted_bound(contexts):
    # Computes the probability weighted lower Hoeffding bound for the expected reward
    # of a list of contexts
    # TODO
    pass


def split_condition(context_1, context_2, base_context):
    # Returns a tuple containing the marginal difference for the split between contexts
    # w.r.t. the base_context
    # Use split_condition >= 0 to obtain the corresponding truth value
    return (
        context_1.weighted_bound
        + context_2.weighted_bound
        - base_context.weighted_bound
    )


def feature_split(dataset, context: Context, feature):
    # Generates a tuple containing two new contexts given a splitting feature
    # TODO
    pass


def tree_generation(dataset, features):
    # Get the base context data
    base_context, _ = feature_split(dataset)
    # Start generating tree
    return generate_tree_node(dataset, features, base_context)


def generate_tree_node(dataset, unsplit_features, base_context):
    # Generate all the possible splits for a single feature
    split_contexts = [
        feature_split(dataset, base_context, feature) for feature in unsplit_features
    ]
    # Map the split contexts over a tuple containing the marginal difference of the split
    # condition and the related contexts
    split_contexts = list(
        map(
            lambda t_contexts: (
                t_contexts,
                split_condition(*(t_contexts), base_context),
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
        # Compute the new set of features
        new_features = list(unsplit_features)
        new_features.remove(best_split[0].last_feature)

        # Helper function to map recursive call results
        def none_context(context, result):
            return result if result else context

        # Call recursively the tree function and create tuples with the
        ret_contexts = list(
            map(
                lambda context: none_context(
                    generate_tree_node(dataset, new_features, context), context
                ),
                best_split,
            )
        )
        # Return flattened list
        return list(itertools.chain(*ret_contexts))
    return None
