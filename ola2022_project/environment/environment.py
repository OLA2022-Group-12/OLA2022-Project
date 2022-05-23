import enum
from collections import namedtuple
import numpy as np
from numpy.random import default_rng
from dataclasses import dataclass, asdict
from typing import Optional

"""The correct use of this module is to construct the class
EnvironmentData by using the function example_environment which returns an
instance of EnvironmentData with sample values. The class can also be created
by itself by specifying all attributes. Then the `get_day_of_interactions`
function should be called to generate interactions.
"""


# Named tuple containing the fundamental parameters unique to each class
UserClassParameters = namedtuple(
    "UserClassParameters", ["reservation_price", "steepness", "shift", "upper_bound"]
)

# Named tuple containing the outcome of a user interaction
Interaction = namedtuple("Interaction", ["user_class", "items_bought"])


class Step(enum.Enum):
    ONE = ("classes_parameters",)
    TWO = ("classes_parameters", "graph")


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
    class_ratios: list[float]

    # Price of the 5 products
    product_prices: list[float]

    # List of class parameters for each class, implemented as list of UserClassParameters
    classes_parameters: list[UserClassParameters]

    # The competitor budget is assumed to be constant, since the competitor is
    # non-strategic
    competitor_budget: int

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
    next_products: list[tuple[int, int]]


@dataclass
class MaskedEnvironmentData:

    """Dataclass containing environment values which are not masked for the
    current learning step See EnvironmentData for more information on the
    different fields."""

    # The total budget to subdivide
    total_budget: int

    # Price of the 5 products
    product_prices: list[float]

    # Lambda parameter, which is the probability of osserving the next secondary product
    # according to the project's assignment
    lam: float

    # List that constains for every i+1 product the secondary i+1 products that will be shown
    # in the first and second slot
    next_products: list[tuple[int, int]]

    # The competitor budget is assumed to be constant, since the competitor is
    # non-strategic
    competitor_budget: Optional[int] = None

    # Max number of items a customer can buy of a certain product. The number of
    # bought items is determined randomly with max_items as upper bound
    max_items: Optional[int] = None

    # Products graph's matrix. It's a empty matrix, should be initialized with populate_graphs
    graph: Optional[np.ndarray] = None

    # Probability of every class to show up. They must add up to 1
    class_ratios: Optional[list[float]] = None

    # List of class parameters for each class, implemented as list of UserClassParameters
    classes_parameters: Optional[list[UserClassParameters]] = None


def create_masked_environment(
    step: Step, env: EnvironmentData
) -> MaskedEnvironmentData:
    filtered_data = asdict(env)
    for name in step.value:
        del filtered_data[name]

    return MaskedEnvironmentData(**filtered_data)


def example_environment(
    rng=default_rng(),
    total_budget=300,
    class_ratios=[0.3, 0.6, 0.1],
    product_prices=[10, 15, 25, 18, 5],
    classes_parameters=[
        UserClassParameters(10, 0.17, 4.5, 50),
        UserClassParameters(20, 0.15, 5, 65),
        UserClassParameters(30, 0.22, 5.3, 100),
    ],
    competitor_budget=100,
    lam=0.5,
    max_items=3,
    graph_fully_connected=True,
    graph_zeros_probability=0.5,
    next_products=[(2, 3), (0, 2), (1, 4), (4, 1), (3, 0)],
):

    """Creates an environment with sample data. Single arguments can be
    specified if desired.

    Arguments:
        rng: numpy generator (such as default_rng)

        total_budget: The total amount of budget to subdivide

        class_ratios: ratios in which every class appears in the population

        product_prices: list containing prices for every i+1 product

        classes_parameters: list containing UserClassParameters for every user class
            we have

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
    graph = generate_graph(
        rng, len(product_prices), graph_fully_connected, graph_zeros_probability
    )

    return EnvironmentData(
        total_budget=total_budget,
        class_ratios=class_ratios,
        product_prices=product_prices,
        classes_parameters=classes_parameters,
        competitor_budget=competitor_budget,
        lam=lam,
        max_items=max_items,
        graph=graph,
        next_products=next_products,
    )


def alpha_function(budget, steepness, shift, upper_bound):

    """Alpha function with the shape of a sigmoidal function. Computes the
    expected value of clicks given a certain budget.

    Arguments:
        budget: integer or float representing the budget

        steepness: float representing the steepness of the sigmoidal curve

        shift: float representing the shift of the entire function. The shift
            should always be positive and not smaller than 3 circa

        upper_bound: integer representing the maximum expected number of \
            clicks possible

    Returns:
        A float representing the expected value of number of clicks for a
        certain class function
    """

    return upper_bound * (1 / (1 + np.exp(-steepness * budget + shift)))


def generate_graph(
    rng: np.random.Generator, size: int, fully_connected: bool, zeros_probability: float
):

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
            [True, False],
            size=(size, size),
            p=[zeros_probability, 1 - zeros_probability],
        )  # type: ignore
        graph[mask] = 0

    # Removes auto-loops from graph
    np.fill_diagonal(graph, 0)

    return graph


def get_day_of_interactions(
    rng: np.random.Generator,
    num_customers: int,
    budgets: np.ndarray,
    env_data: EnvironmentData,
) -> list[Interaction]:

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

    Returns:
        A list of interactions generated based on the given arguments
    """
    n_classes = 3
    n_products = 5

    # We will traverse the graph AT MOST n_products times per customer, as we
    # cannot buy a product multiple times, hence it is enough to generate
    # n_products uniform variables for each customer to calculate the next
    # product probability.
    next_product_probabilities = rng.uniform(
        low=0.0, high=1.0, size=(num_customers, n_products)
    )

    # IF we decide to buy product i for customer j, we will buy this amount of
    # it
    items_of_product = rng.integers(
        1, env_data.max_items, endpoint=True, size=(num_customers, n_products)
    )

    # Competitor budget is added to the array of budgets at the end
    budgets = np.append(budgets, env_data.competitor_budget)

    # The ratios of users assigned to each class given the parameters of the
    # classes function
    alpha_ratios = np.array(
        [
            rng.dirichlet(
                alpha_function(
                    budgets,
                    class_params.steepness,
                    class_params.shift,
                    class_params.upper_bound,
                )
            )
            for class_params in env_data.classes_parameters
        ]
    )

    # By multipling the matrix of the alpha_ratios and the class ratios, we get
    # a matrix which combines both probability distributions. This means that
    # the total sum of the entire matrix is still 1, and by "flattening" it we
    # can use it directly into the rng.choice call as the uneven probability
    # distribution. The benefit is that rng.choice is probably better at
    # choosing evenly that any rounding we would do manually.
    product_and_class_probabilities = (
        alpha_ratios.T * np.array(env_data.class_ratios)
    ).T

    # This is just the possibilities that can be chosen, which is a direct
    # mapping of the probability distribution above
    product_and_class_possible_choices = np.mgrid[: n_products + 1, :n_classes].T
    product_and_class_choices = rng.choice(
        product_and_class_possible_choices.reshape(-1, 2),
        p=product_and_class_probabilities.reshape(-1),
        size=num_customers,
    )

    total_interactions = []
    for (
        product_and_class,
        items_of_product_for_interaction,
        next_product_probabilities_for_interaction,
    ) in zip(
        product_and_class_choices,
        items_of_product,
        next_product_probabilities,
    ):
        main_product = product_and_class[0]
        user_class = product_and_class[1]

        # Competitor, so does not generate a Interaction
        if main_product == n_products:
            continue

        # Simple BFS through the graph, which only continues if we actually buy
        # elements at the found product
        bought_products = set()
        pages = [(main_product, env_data.next_products[main_product])]
        while len(pages) > 0:
            product, slots = pages.pop()

            if (
                env_data.product_prices[product]
                > env_data.classes_parameters[user_class].reservation_price
            ):
                # We pass this product as the price is too high, don't look at
                # neither slots, but there might be more pages left to look at.
                continue

            # We buy the current product as the customer can afford it
            bought_products.add(product)

            for i, product_in_slot in enumerate(slots):
                if product_in_slot in bought_products:
                    # We already bought the product in this slot
                    continue

                if (
                    next_product_probabilities_for_interaction[product_in_slot]
                    < env_data.graph[product, product_in_slot] * (env_data.lam**i)
                    and product_in_slot not in bought_products
                ):
                    # We chose the product in slot i, so we have to add it's
                    # "page" to the pages we need to check
                    pages.append(
                        (product_in_slot, env_data.next_products[product_in_slot])
                    )

        if len(bought_products) == 0:
            continue

        items_bought = np.zeros_like(items_of_product_for_interaction)
        for product in bought_products:
            items_bought[product] = items_of_product_for_interaction[product]

        total_interactions.append(Interaction(user_class, items_bought))

    return total_interactions
