import itertools
from dataclasses import dataclass
from typing import List, Optional
from ola2022_project.utils import compute_hoeffding_bound


@dataclass
class Context:

    """Dataclass containing all the information related to contexts used
    in contextual generation. Its main purpose is to facilitate feature
    splitting algorithms by providing an interface capable of enclosing
    all the relevant information for choosing and evaluating different
    scenarios.
    """

    # List of current context features
    features: List[int]

    # Last added feature stored as an index for feature list
    last_feature: Optional[int]

    # Expected appearance probabilities for each feature that is relevant to the context
    exp_probs: List[float]

    # Number of samples found in the training dataset for each feature that is
    # relevant to the context
    nums: List[int]

    # Maximum expected reward
    max_exp_reward: float

    # Probability-weighted lower Hoeffding bound for the maximum expected reward
    weighted_bound: float


def compute_weighted_bound(ctx: Context) -> float:

    """Computes the probability-weighted lower Hoeffding bound for the expected reward
    of a given context.

    Arguments:
        ctx: context under examination

    Returns:
        Floating value corresponding to the weighted lower bound
    """

    p = ctx.exp_prob[ctx.last_feature] if ctx.last_feature else 1
    mu = ctx.max_exp_reward
    return (p - compute_hoeffding_bound(p)) * (mu - compute_hoeffding_bound(mu))


def split_condition(
    context_1: Context, context_2: Context, base_context: Context
) -> float:

    """Computes the marginal difference for the split between contexts w.r.t. the base_context,
    often utilized to determined if a given feature split is worth.

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


def feature_split(dataset, context: Context, feature=None) -> List[Context]:

    """Generates, trains and evaluates over a given dataset new contexts based on a feature split,
    starting from a base context and resulting in a binary split, therefore two new contexts with
    a dependency on the given feature. If the context and the feature are not present, the function
    generates a single new base aggregated context over the dataset.

    Arguments:
        dataset: current offline dataset gathered over a span of time containing samples
            used to train new models and define context attributes

        context: context utilized as a starting base for the feature split

        feature: feature that is being split upon

    Returns:
        A list of resulting contexts that may contain one to two elements based on the
        given arguments
    """

    pass  # TODO


def tree_generation(dataset, features: List[int]) -> List[Context]:

    """Computes the feature tree for a given dataset and set of features, it utilizes a
    greedy approach over feature splits by iteratively splitting on the most promising
    feature for each tree level; the splitting condition takes into consideration the
    lower bounds for the deciding factors since a pessimistic approach is needed in order
    to control the costly operation of splitting a context into sub-contexts. Being a greedy
    approach, the result holds no guarantee of optimality but underlines the most important
    features of the dataset effectively.

    Arguments:
        dataset: current offline dataset gathered over a span of time, containing samples
            used to train new models and define context attributes

        features: list of all the possible features of the samples contained in the dataset

    Returns:
        A list containing the contexts chosen by the algorithm
    """

    # Get the base context data
    base_context = feature_split(dataset)
    # Start generating tree recursively
    optimal_contexts = generate_tree_node(dataset, features, base_context)
    return optimal_contexts if optimal_contexts else base_context


def generate_tree_node(
    dataset, unsplit_features, base_context
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
            lambda feature: feature_split(dataset, base_context, feature),
            unsplit_features,
        )
    )
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
