import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from ola2022_project.utils import compute_hoeffding_bound, flatten_list
from ola2022_project.environment.environment import Feature
from ola2022_project.simulation.simulation import Simulation


logger = logging.getLogger(__name__)


@dataclass
class Context:

    """Dataclass containing all the information related to contexts used
    in contextual generation. Its main purpose is to facilitate feature
    splitting algorithms by providing an interface capable of enclosing
    all the relevant information for choosing and evaluating different
    scenarios.
    """

    # Simulation containing the training environment for the learner
    learner_sim: Simulation

    # List of current context features
    features: List[Feature]

    # Number of samples found in the training dataset for each feature that is
    # relevant to the context
    nums: int

    # Expected appearance probabilities for the feature values identified by the context
    exp_prob: float

    # Maximum expected reward
    max_exp_reward: float

    # Probability-weighted lower Hoeffding bound for the maximum expected reward
    weighted_bound: float


def _none_context(context: Context, result: Optional[List[Context]]):

    """Helper function to map recursive calls to node generation.

    Arguments:

        context: context that generated the result

        result: result from feature splitting on a context

    Returns:
        Returns the result or the generating context in case the result is empty
    """

    return result if result else context


def _compute_weighted_bound(np: int, nmu: int, p: float, mu: float) -> float:

    """Computes the probability-weighted lower Hoeffding bound for the expected reward
    of a given probability-expected_reward pair.

    Arguments:

        np: number of samples over which the empiric expected probability was calculated upon

        nmu: number of samples over which the empiric expected reward was calculated upon

        p: empiric expected probability

        mu: empiric expected reward


    Returns:
        Floating value corresponding to the weighted lower bound
    """

    return (p - compute_hoeffding_bound(np)) * (mu - compute_hoeffding_bound(nmu))


def _split_condition(
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


def _feature_split(
    sim_model: Simulation,
    dataset,
    context: Context = None,
    feature: Feature = None,
) -> List[Context]:

    """Generates, trains and evaluates over a given dataset new contexts based on a feature split,
    starting from a base context and resulting in a binary split, therefore it creates two new
    contexts with a dependency on the given feature. If the context or the feature are not
    present, the function generates a single new base aggregated context over the dataset.

    Arguments:
        sim_model: simulation model containing the parameters that will be used by the dataset
            simulation to operate the learner and the environment

        dataset: current offline dataset gathered over a span of time containing samples
            used to train new models and define context attributes

        context: context utilized as a starting base for the feature split

        feature: feature that is being split upon

    Returns:
        A list of resulting contexts that may contain one to two elements based on the
        given arguments
    """

    if not context or not feature:
        # Base case for the root of the decision tree, where all features are aggregated
        # and no split is needed
        sim_model.simulate(len(dataset))
        n = sum([len(d) for d in sim_model.dataset])
        # TODO: max expected reward (temporary solution: max reward over dataset)
        max_exp_reward = np.max(sim_model.rewards)

        logger.debug("Root context")
        logger.debug(f"Maximum expected reward {max_exp_reward}")

        return Context(
            sim_model,
            [],  # In the base model all features are aggregated
            n,
            1,  # Probability = 100%
            max_exp_reward,
            _compute_weighted_bound(n, n, 1, max_exp_reward),
        )

    # Split feature given as parameter
    feature_1 = Feature(feature.name, 0)
    feature_2 = Feature(feature.name, 1)

    # Return the two contexts, each evaluating a value of the split feature
    return [
        _feature_half_split(sim_model.copy(), dataset, context, feature_1),
        _feature_half_split(sim_model.copy(), dataset, context, feature_2),
    ]


def _feature_half_split(
    sim_model: Simulation, dataset, context: Context, feature: Feature
) -> Context:

    """Generates, trains and evaluates over a given dataset a new context based on half of a feature
    binary split.

    Arguments:
        sim_model: simulation model containing the parameters that will be used by the dataset
            simulation to operate the learner and the environment

        dataset: current offline dataset gathered over a span of time containing samples
            used to train new models and define context attributes

        context: context utilized as a starting base for the feature split

        feature: feature that is being split upon

    Returns:
        A new context trained and evaluated over the filtered dataset
    """

    # Compute resulting set of features
    features = context.features.copy()
    features.append(feature)

    # Create and train a new context
    new_context = Context(sim_model, features, 0, 1, 0, 0)
    train_context(new_context, dataset)

    return new_context


def train_context(context: Context, dataset, update: bool = True):

    """Trains and evaluates a context by simulating interactions utilizing a dataset as a reference;
    the context may be new or already trained, in the latter case only the difference in days
    between the dataset and the context history is taken into consideration.

    Arguments:
        context: the context that will be trained and evaluated

        dataset: current offline dataset gathered over a span of time containing samples
            used to train new models and define context attributes

        update: if False, the learner won't gain any additional knowledge from the dataset
    """

    new_days = len(dataset) - len(context.learner_sim.dataset)

    # Training simulation
    context.learner_sim.simulate(new_days, context.features, update=update)

    # Count total number of samples for datasets and expected probability of a sample
    # presenting the features of interest
    if context.features:
        context.nums = sum([len(d) for d in context.learner_sim.filtered_dataset])
        context.exp_prob = context.nums / sum(
            [len(d) for d in context.learner_sim.dataset]
        )
    else:
        context.nums = sum([len(d) for d in context.learner_sim.dataset])
        context.exp_prob = 1

    # Reward simulation
    reward_sim = context.learner_sim.copy()
    reward_sim.learner = context.learner_sim.learner
    reward_sim.simulate(len(dataset), update=False)

    # Compute maximum expected reward and count samples
    n_reward = sum([len(d) for d in reward_sim.dataset])
    # TODO: max expected reward (temporary solution: max reward over dataset)
    context.max_exp_reward = np.max(reward_sim.rewards)

    # Compute the weighted bound
    context.weighted_bound = _compute_weighted_bound(
        context.nums, n_reward, context.exp_prob, context.max_exp_reward
    )

    print(f"Context features {context.features}")
    print(f"Expected probability {context.exp_prob}")
    print(f"Maximum expected reward {context.max_exp_reward}")
    print(f"Weighted lower bound {context.weighted_bound}")


def tree_generation(
    sim_reference: Simulation, dataset, features: List[Feature]
) -> List[Context]:

    """Computes the feature tree for a given dataset and set of features, it utilizes a
    greedy approach over feature splits by iteratively splitting on the most promising
    feature for each tree level; the splitting condition takes into consideration the
    lower bounds for the deciding factors since a pessimistic approach is needed in order
    to control the costly operation of splitting a context into sub-contexts. Being a greedy
    approach, the result holds no guarantee of optimality but underlines the most important
    features of the dataset effectively.

    Arguments:
        sim_reference: simulation reference from which the parameters will be copied and used
            to construct new simulations that operate on the dataset

        dataset: current offline dataset gathered over a span of time, containing samples
            used to train new models and define context attributes

        features: list of all the possible features of the samples contained in the dataset

    Returns:
        A list containing the contexts chosen by the algorithm
    """

    # Obtain the base context at the root of the tree
    base_context = _feature_split(sim_reference.copy(), dataset)
    # Start generating tree recursively
    optimal_contexts = _generate_tree_node(
        sim_reference.copy(), dataset, features, base_context
    )
    return optimal_contexts if optimal_contexts else base_context


def partial_tree_generation(
    dataset,
    features: List[Feature],
    root: List[Context],
) -> List[Context]:

    """Performs the tree generation algorithm from a starting point represented by a set
    of contexts.

    Arguments:
        dataset: current offline dataset gathered over a span of time, containing samples
            used to train new models and define context attributes

        features: list of all the possible splittable features of the samples contained
            in the dataset

        root: set of contexts that define the starting point of the algorithm

    Returns:
        A list containing the contexts chosen by the algorithm
    """

    # Update experimental rewards of old base contexts
    for context in root:
        train_context(context, dataset, update=False)

    ret_contexts = list(
        map(
            lambda context: _none_context(
                context,
                _generate_tree_node(
                    context.learner_sim.copy(), dataset, features, context
                ),
            ),
            root,
        )
    )
    return flatten_list(ret_contexts)


def _generate_tree_node(
    sim_model: Simulation,
    dataset,
    unsplit_features: List[Feature],
    base_context: Context,
    exp: bool = True,
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
        sim_model: simulation model containing the parameters that will be used by the dataset
            simulation to operate the learner and the environment

        dataset: current offline dataset gathered over a span of time, containing samples
            used to train new models and define context attributes

        unsplit_features: list of features that have yet to be split upon

        base_context: reference base context for a split

    Returns:
        A list of "greedy optimal" contexts for the subproblem
    """

    # If there are no more features to split, stop generating the tree
    if not unsplit_features:
        return None

    # Generate all the possible splits for a single feature
    split_contexts = list(
        map(
            lambda feature: _feature_split(
                sim_model.copy(), dataset, base_context, feature
            ),
            unsplit_features,
        )
    )
    # Map the split contexts over a tuple containing the marginal difference of the split
    # condition and the related contexts
    split_contexts = list(
        map(
            lambda contexts: (
                contexts,
                _split_condition(*(contexts), base_context),
            ),
            split_contexts,
        )
    )
    # Filter the split contexts based on the split_condition in order to only evaluate the
    # ones that are worth splitting
    # The split_condition is on the right side of the tuple, hence, it is accessed by using [1]
    split_contexts = list(
        filter(lambda t_contexts_split: t_contexts_split[1] >= 0, split_contexts)
    )

    if split_contexts:
        # Take the split with best marginal value
        best_split, _ = max(
            split_contexts, key=lambda t_contexts_split: t_contexts_split[1]
        )
        # Compute the new set of unsplit features by adding the last feature added to one of the
        # contexts (which one is indifferent, here we chose the first one: index 0)
        new_features = [
            feature
            for feature in unsplit_features
            if feature.name != best_split[0].features[-1].name
        ]

        # Call recursively the tree function and create tuples
        ret_contexts = list(
            map(
                lambda context: _none_context(
                    context,
                    _generate_tree_node(sim_model, dataset, new_features, context, exp),
                ),
                best_split,
            )
        )
        # Return flattened list
        return (
            flatten_list(ret_contexts)
            if isinstance(ret_contexts, list)
            else [ret_contexts]
        )
    return None
