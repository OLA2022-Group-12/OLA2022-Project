import logging
from aenum import Enum, NoAlias

from typing import Optional, List, Tuple
from collections import namedtuple
from dataclasses import dataclass, asdict
import numpy as np
from numpy.random import default_rng
from ola2022_project.utils import replace_zeros

"""The correct use of this module is to construct the class
Environment_data by using the function example_environment which returns an
instance of Environment_data with sample values. The class can also be created
by itself by specifying all attributes.
"""

logger = logging.getLogger(__name__)


# Named tuple containing the fundamental parameters unique to each class
UserClassParameters = namedtuple(
    "UserClassParameters", ["reservation_price", "upper_bound", "max_useful_budget"]
)

# Contains the name and the value of a single user feature (which can be either 0 or 1)
Feature = namedtuple("Feature", ["name", "value"])

# Named tuple containing the outcome of a user interaction
Interaction = namedtuple(
    "Interaction", ["user_features", "user_class", "items_bought", "landed_on", "edges"]
)

# Similar to Interaction but doesn't cointain any reference to a user class
AggregatedInteraction = namedtuple(
    "AggregatedInteraction", ["items_bought", "landed_on"]
)

# Used to define how interactions should be generated. Basically instead of representing
# a single interaction it tells how a group of users will behave
InteractionBlueprint = namedtuple(
    "InteractionBlueprint", ["user_class", "landed_on", "n_users", "products", "path"]
)


class Step(Enum):

    _settings_ = NoAlias

    CLAIRVOYANT = ()
    ZERO = ()
    ONE = ("classes_parameters",)
    TWO = ("classes_parameters",)
    THREE = ("graph",)


@dataclass
class EnvironmentData:

    """Dataclass containing environment values. Should be constructed passing
    the rng parameter, which is the only one not defined by default. The other
    parameters are set up correctly by the dataclass constructor, with default
    values as shown below. Setups correctly everything about the environment.
    After constructing this class, it can be passed to the function
    get_day_of_interactions. To construct it with different values they can
    simply be specified when the class is constructed.
    """

    # The total budget to subdivide
    total_budget: int

    # Probability of every class to show up. They must add up to 1
    class_ratios: List[float]

    # Features associated to every class
    class_features: List[List]

    # Price of the 5 products
    product_prices: List[float]

    # List of class parameters for each class and product, implemented as list
    # of lists of UserClassParameters. Each class has distinct parameters for
    # every product
    classes_parameters: List[List[UserClassParameters]]

    # Lambda parameter, which is the probability of osserving the next secondary product
    # according to the project's assignment
    lam: float

    # Max number of items a customer can buy of a certain product. The number of
    # bought items is determined randomly with max_items as upper bound
    max_items: int

    # Products graph's matrix. It's a empty matrix, should be initialized with populate_graphs
    graph: np.ndarray

    # List that constains for every i+1 product the secondary i+1 products that will be shown
    # in the first and second slot
    next_products: List[Tuple[int, int]]

    # Controls randomness of the environment
    random_noise: float


@dataclass
class MaskedEnvironmentData:

    """Dataclass containing environment values which are not masked for the
    current learning step See EnvironmentData for more information on the
    different fields."""

    # The total budget to subdivide
    total_budget: int

    # Price of the 5 products
    product_prices: List[float]

    # Lambda parameter, which is the probability of osserving the next secondary product
    # according to the project's assignment
    lam: float

    # List that constains for every i+1 product the secondary i+1 products that will be shown
    # in the first and second slot
    next_products: List[Tuple[int, int]]

    # Max number of items a customer can buy of a certain product. The number of
    # bought items is determined randomly with max_items as upper bound
    max_items: Optional[int] = None

    # Products graph's matrix. It's a empty matrix, should be initialized with populate_graphs
    graph: Optional[np.ndarray] = None

    # Probability of every class to show up. They must add up to 1
    class_ratios: Optional[List[float]] = None

    # Features associated to every class
    class_features: Optional[List[List]] = None

    # List of class parameters for each class and product, implemented as list
    # of lists of UserClassParameters. Each class has distinct parameters for
    # every product
    classes_parameters: Optional[List[List[UserClassParameters]]] = None

    # Controls randomness of the environment
    random_noise: float = None


def create_masked_environment(
    step: Step, env: EnvironmentData
) -> MaskedEnvironmentData:
    filtered_data = asdict(env)
    for name in step.value:
        try:
            del filtered_data[name]
        except KeyError:
            logger.debug(
                f"Programming error: Step configured with field not found in environment: '{name}'"
            )

    return MaskedEnvironmentData(**filtered_data)


def example_environment(
    rng=default_rng(),
    total_budget=300,
    class_ratios=[0.3, 0.6, 0.1],
    class_features=[
        [
            [Feature("feature_1", 0), Feature("feature_2", 0)],
            [Feature("feature_1", 0), Feature("feature_2", 1)],
        ],
        [[Feature("feature_1", 1), Feature("feature_2", 0)]],
        [[Feature("feature_1", 1), Feature("feature_2", 1)]],
    ],
    product_prices=[3, 15, 8, 22, 1],
    classes_parameters=[
        [
            UserClassParameters(10, 0.2, 120),
            UserClassParameters(10, 0.15, 120),
            UserClassParameters(8, 0.5, 300),
            UserClassParameters(7, 0.05, 220),
            UserClassParameters(14, 0.15, 170),
        ],
        [
            UserClassParameters(22, 0.5, 190),
            UserClassParameters(20, 0.1, 210),
            UserClassParameters(16, 0.25, 240),
            UserClassParameters(24, 0.03, 80),
            UserClassParameters(20, 0.05, 360),
        ],
        [
            UserClassParameters(33, 0.4, 180),
            UserClassParameters(25, 0.15, 210),
            UserClassParameters(30, 0.35, 240),
            UserClassParameters(31, 0.05, 300),
            UserClassParameters(36, 0.05, 420),
        ],
    ],
    lam=0.5,
    max_items=3,
    graph_fully_connected=True,
    graph_zeros_probability=0.5,
    next_products=[(2, 3), (0, 2), (1, 4), (4, 1), (3, 0)],
    random_noise=1e-2,
    graph=np.array(
        [
            [0, 0, 0.7, 0.4, 0],
            [0.3, 0, 0.8, 0, 0],
            [0, 0.2, 0, 0, 0.2],
            [0, 0.9, 0, 0, 0.8],
            [0.05, 0, 0, 0.25, 0],
        ]
    ),
):

    """Creates an environment with sample data. Single arguments can be
    specified if desired.

    Arguments:
        rng: numpy generator (such as default_rng)

        total_budget: The total amount of budget to subdivide

        class_ratios: ratios in which every class appears in the population

        product_prices: list containing prices for every i+1 product

        classes_parameters: list containing UserClassParameters for every user class
            we have and every product (including the competitor)

        competitor_budget: budget the competitor spends in advertising

        lam: lambda value representing the chance for a user to look at the
            second product slot after looking at the first slot

        max_items: maximum number of items a user can buy of a certain product

        graph_fully_connected: boolean value asserting if the product graph
            should be generated fully connected

        graph_zeros_probability: float representing the chance to find a missing
            link in a not fully connected graph

        next_products: list of tuples. Every i+1 products can show as secondary
            products the ones in the i-tuple, in the first and second slot
            respectively

    Returns:
        An EnvironmentData instance, containing the sample values
    """

    # Graph is generated accoring to indications
    if graph is None:
        graph = generate_graph(
            rng, len(product_prices), graph_fully_connected, graph_zeros_probability
        )

    return EnvironmentData(
        total_budget=total_budget,
        class_ratios=class_ratios,
        class_features=class_features,
        product_prices=product_prices,
        classes_parameters=classes_parameters,
        lam=lam,
        max_items=max_items,
        graph=graph,
        next_products=next_products,
        random_noise=random_noise,
    )


def alpha_function(budget, upper_bound, max_useful_budget):

    """Alpha function with the shape of a exponential function. Computes the
    expected value of clicks given a certain budget.

    Arguments:
        budget: integer or float representing the budget

        upper_bound: integer representing the maximum expected number of
            clicks possible

        max_useful_budget: maximum amount of budget after which increasing
            the budget won't lead to a ratio increase

    Returns:
        A float representing the expected value of number of clicks for a
        certain class function
    """

    if max_useful_budget > 2e-16:
        steepness = 4 / max_useful_budget
        return upper_bound * (1 - np.exp(-steepness * budget))

    # In this case max_useful_budget == 0 circa, this means that we
    # immediately reach the upper bound
    else:
        return upper_bound


def generate_graph(rng, size, fully_connected, zeros_probability):

    """This function generates by default a incidence matrix of a graph

    Arguments:
        rng: instance of a generator (default_rng())

        size: integer corresponding to the number of nodes

        fully_connected: boolean value. If set to true, function will generate a
            fully connected graph, otherwise if set to false the graph won't be
            fully connected, it will miss some connections

        zeros_probability: number between 0 and 1 which represents probability
            to not have a connection between two nodes. Only has effect if
            fully_connected is set to False

    Returns:
        A numpy square matrix (size x size) where every element (i,j) represents
        the weight of the conncetion between node-i and node-j
    """

    # Generates a random fully-connected graph
    graph = rng.random((size, size))

    # Removes some connections if graph is requested to be not fully-connected
    if not fully_connected:
        mask = rng.choice(
            [True, False], (size, size), p=[zeros_probability, 1 - zeros_probability]
        )
        graph[mask] = 0

    # Removes auto-loops from graph
    np.fill_diagonal(graph, 0)

    return graph


def get_day_of_interactions(
    rng: np.random.default_rng,
    population,
    budgets,
    env_data: EnvironmentData,
    de_noise=1e3,
    deterministic=False,
):

    """Main method to be called when interacting with the environment. Outputs
    all the interactions of an entire day. When called generates new alphas from
    the clicks function with budget as input.

    Arguments:
        rng: instance of a generator (default_rng())

        num_customers: total number of customers that will make interactions in
            a day

        budgets: numpy array of 5 integers containing the budget for each
            product. The i-element is the budget for the i+1 product

        env_data: instance of Environment_data

        de_noise: controls the amount of noise of the Dirichlet functions.
            The higher the value, the lower the noise will be.

        deterministic: if set to True all the interactions will be generated in a
            deterministic way. Mind that in this case every user will buy at most
            one item per product

    Returns:
        A list of tuples, with a size corresponding to num_customers. Every
        tuple denotes an interaction. tuple[0] is the user's class, which can be
        1, 2 or 3. tuple[1] is a numpy array of 5 elemets, where every element i
        represents how many of the products i+1 the customer bought
    """

    n_classes = len(env_data.class_ratios)

    # Computing total number of customers for each class based on class_ratios
    customers_per_class = [
        int(np.rint(population * ratio)) for ratio in env_data.class_ratios
    ]

    logger.debug(
        f"Population divided between classes as follows: {customers_per_class}"
    )

    # If the budgets array is 1-dimensional it means that we are optimizing for a
    # single context (no splitting has happened)
    if len(np.shape(budgets)) == 1:
        budget_allocation = np.array(
            [np.array(budgets) / n_classes for _ in range(n_classes)]
        )
        logger.debug("Targeting 1 context in current environment")

    # If the array is 2-dimensional it means that we are optimizing for more than
    # one context
    # TODO implement this
    elif len(np.shape(budgets)) == 2:
        raise RuntimeError(
            "Cannot handle multiple contexts, still has to be implemented"
        )

    else:
        raise RuntimeError(f"Invalid budget shape: {np.shape(budgets)}")

    # total_interactions = list()
    interaction_blueprints = []

    # Generate interactions for every class
    for user_class, (class_population, class_parameters, assigned_budget) in enumerate(
        zip(customers_per_class, env_data.classes_parameters, budget_allocation)
    ):

        alpha_ratios = [
            alpha_function(product_budget, params.upper_bound, params.max_useful_budget)
            for params, product_budget in zip(class_parameters, assigned_budget)
        ]

        competitor_ratio = 1 - np.sum(alpha_ratios)

        if competitor_ratio < 0:
            raise RuntimeError("Bad alpha-function parameters, total ratio ecceeds 1")

        alpha_ratios.append(competitor_ratio)

        logger.debug(f"Computed alpha ratios: {alpha_ratios}")

        # Replace ratios that are 0 with machine-espilon (10^-16) to ensure
        # compatibility with the Dirichlet function
        alpha_ratios = replace_zeros(alpha_ratios)

        if not deterministic:
            alpha_ratios_noisy = rng.dirichlet(np.array(alpha_ratios) * de_noise)

        else:
            alpha_ratios_noisy = np.array(alpha_ratios)

        users_landing_on_pages = np.rint(
            np.delete(alpha_ratios_noisy, -1) * class_population
        ).astype(int)

        logger.debug(
            f"Dirichlet output (doesn't include competitor): {alpha_ratios_noisy}"
        )

        # According to product ratios, for every product the computed number of
        # users lands on the correct product and the interaction starts
        for product, n_users in enumerate(users_landing_on_pages):

            products = np.zeros(len(env_data.product_prices), dtype=int)
            path = []
            interaction_blueprints = _simulate_interaction(
                rng,
                env_data,
                n_users,
                user_class,
                products,
                product,
                path,
                interaction_blueprints,
                product,
                deterministic,
            )

    # Generate interactions starting from blueprints and shuffle them to make
    # data more realistic
    total_interactions = _generate_interactions(
        interaction_blueprints, env_data, rng, deterministic
    )
    rng.shuffle(total_interactions)
    return total_interactions


def _simulate_interaction(
    rng,
    env_data: EnvironmentData,
    n_clicks,
    user_class,
    products_bought,
    primary_product,
    path,
    interaction_blueprints,
    first_product,
    deterministic,
) -> List[InteractionBlueprint]:

    """Simulate interactions of a primary product page based on the given
    parameters and outputs blueprints for every possible interaction.

    Arguments:
        rng: an instance of numpy.random.default_rng()

        env_data: an instance of EnvironmentData

        n_clicks: number of users that landed on the specified primary product

        user_class: class of which the users belong to

        products_bought: one-hot encoded array keeping track of the products bought
            by the specified users

        path: list of activated graph edges up until now

        interaction_blueprints: list of interactions blueprint generated up until now

        first_product: first product the group of users landed on

        deterministic: if set to True the generated interactions will be deterministic

    Returns:
        A list of InteractionBlueprint
    """

    if not deterministic:
        random_noise = env_data.random_noise

    else:
        random_noise = 0

    product_price = env_data.product_prices[primary_product]

    # If the price is under the reservation price of the given class for this particlar
    # product, the group of users will buy it
    if (
        env_data.classes_parameters[user_class][primary_product].reservation_price
        >= product_price
    ):
        products_bought[primary_product] = 1

        slot1 = env_data.next_products[primary_product][0]
        slot2 = env_data.next_products[primary_product][1]

        # We make fresh copies of the activated edges and the products_bought because
        # these will be used to generate new blueprints representing users clicking
        # on one of the suggested products
        slot1_path = path.copy()
        slot1_path.append((primary_product, slot1))
        slot1_products_bought = products_bought.copy()

        slot2_path = path.copy()
        slot2_path.append((primary_product, slot2))
        slot2_products_bought = products_bought.copy()

        # If the group of users has not bought the suggested item, a percentage
        # of them will click on it
        n_users_click_slot_1 = 0
        if products_bought[slot1] == 0:
            n_users_click_slot_1 = int(
                np.rint(
                    n_clicks
                    * rng.normal(env_data.graph[primary_product, slot1], random_noise)
                )
            )
            # If the resulting number of users is positive, we generate new blueprints
            # starting from the clicked product
            if n_users_click_slot_1 > 0:
                _simulate_interaction(
                    rng,
                    env_data,
                    n_users_click_slot_1,
                    user_class,
                    slot1_products_bought,
                    slot1,
                    slot1_path,
                    interaction_blueprints,
                    first_product,
                    deterministic,
                )

        # The same is done for the second slot product, keeping in account the lambda parameter
        n_users_click_slot_2 = 0
        if products_bought[slot2] == 0:
            n_users_click_slot_2 = int(
                np.rint(
                    (n_clicks - n_users_click_slot_1)
                    * rng.normal(env_data.graph[primary_product, slot2], random_noise)
                    * env_data.lam
                )
            )
            if n_users_click_slot_2:
                _simulate_interaction(
                    rng,
                    env_data,
                    n_users_click_slot_2,
                    user_class,
                    slot2_products_bought,
                    slot2,
                    slot2_path,
                    interaction_blueprints,
                    first_product,
                    deterministic,
                )

        # The users that did not click on either of the two suggested products will stop
        # at the current page
        n_users_stopping = n_clicks - n_users_click_slot_1 - n_users_click_slot_2

        if n_users_stopping > 0:
            interaction_blueprints.append(
                InteractionBlueprint(
                    user_class, first_product, n_users_stopping, products_bought, path
                )
            )

    else:
        interaction_blueprints.append(
            InteractionBlueprint(
                user_class, first_product, n_clicks, products_bought, path
            )
        )

    return interaction_blueprints


def _generate_interactions(
    blueprints: List[InteractionBlueprint], env: EnvironmentData, rng, deterministic
):

    """This function generates actual real interactions starting from interaction blueprints.

    Arguments:
        blueprints: List of InteractionBlueprint

        env: instance of EnvironmentData

        rng: instance of numpy.random.default_rng()

        deterministic: if set to True, the number of items bought for each product will be
            always 1 if the users buys it

    Returns:
        A list of Interactions representing real single-user interactions according to the
        provided interaction blueprints.
    """

    interactions = []

    for blueprint in blueprints:

        for _ in range(blueprint.n_users):

            items_bought = blueprint.products.copy()

            if not deterministic:
                for i in range(len(items_bought)):
                    if items_bought[i] == 1:
                        items_bought[i] = rng.integers(1, env.max_items, endpoint=True)

            feature_idx = rng.integers(len(env.class_features[blueprint.user_class]))
            interactions.append(
                Interaction(
                    env.class_features[blueprint.user_class][feature_idx],
                    blueprint.user_class,
                    items_bought,
                    blueprint.landed_on,
                    blueprint.path,
                )
            )

    return interactions


def remove_classes(interactions: List[Interaction]) -> List[AggregatedInteraction]:
    return [AggregatedInteraction(e.items_bought, e.landed_on) for e in interactions]


# TODO: polish
def feature_filter(dataset, features: List[Feature]):

    """Filters the elements of a dataset given a set of wanted features.

    Arguments:
        dataset: dataset to filter

        features: features to discriminate

    Returns:
        A new dataset composed only of interactions that satisfy the given features
    """

    filtered_dataset = []
    for dataset_day in dataset:
        # Select only the interactions made by users that respect the features specified
        # in the features parameter, this is done by iterating over every day in the dataset
        # and every interaction in each day
        filtered_dataset.append(
            list(
                filter(
                    lambda interaction: all(
                        (feature in interaction.user_features) for feature in features
                    ),
                    dataset_day,
                )
            )
        )
    return filtered_dataset
