import enum
from collections import namedtuple
import numpy as np
from numpy.random import default_rng
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

"""The correct use of this module is to construct the class
Environment_data by using the function example_environment which returns an
instance of Environment_data with sample values. The class can also be created
by itself by specifying all attributes.
"""


# Named tuple containing the fundamental parameters unique to each class
UserClassParameters = namedtuple(
    "UserClassParameters", ["reservation_price", "steepness", "shift", "upper_bound"]
)

# Named tuple containing the outcome of a user interaction
Interaction = namedtuple("Interaction", ["user_class", "items_bought"])


class Step(enum.Enum):
    ZERO = ()
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
    class_ratios: List[float]

    # Price of the 5 products
    product_prices: List[float]

    # List of class parameters for each class, implemented as list of UserClassParameters
    classes_parameters: List[UserClassParameters]

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
    next_products: List[Tuple[int, int]]


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

    # The competitor budget is assumed to be constant, since the competitor is
    # non-strategic
    competitor_budget: Optional[int] = None

    # Max number of items a customer can buy of a certain product. The number of
    # bought items is determined randomly with max_items as upper bound
    max_items: Optional[int] = None

    # Products graph's matrix. It's a empty matrix, should be initialized with populate_graphs
    graph: Optional[np.ndarray] = None

    # Probability of every class to show up. They must add up to 1
    class_ratios: Optional[List[float]] = None

    # List of class parameters for each class, implemented as list of UserClassParameters
    classes_parameters: Optional[List[UserClassParameters]] = None


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
    product_prices=[3, 15, 8, 22, 1],
    classes_parameters=[
        UserClassParameters(10, 0.06, 1, 150),
        UserClassParameters(20, 0.03, 0.4, 100),
        UserClassParameters(30, 0.07, 0.5, 260),
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

    return np.maximum(0, upper_bound * (1 - np.exp(-steepness * budget + shift)))


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


def _get_interaction(rng, user_class, primary_product, env_data):

    """Computes a single interaction and returns it.

    This method shouldn't be called on its own, but only via the
    get_day_of_interactions() method

    Arguments:
        rng: instance of a generator (default_rng())

        user_class: integer from 1 to 3 representing user's class

        primary_product: integer from 0 to 5 representing product number.
            Note: product 0 is the competitor's

        env_data: instance of Environment_data

    Returns:
        A named tuple Interaction, where the field user_class is an integer (1,
        2 or 3) representing the user class, and the field items_bought is a
        numpy array of 5 elements, where every element i is an integer
        indicating the quantity bought of the product i+1
    """

    # This array is initialized with zeros and represents the quantity of bought
    # items for every product in a page
    items_bought = np.zeros(5, dtype=np.int8)

    # The user goes to the page of the primary_product
    items_bought = _go_to_page(rng, user_class, primary_product, items_bought, env_data)

    return user_class, items_bought


def get_day_of_interactions(rng, num_customers, budgets, env_data):

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
        A list of tuples, with a size corresponding to num_customers. Every
        tuple denotes an interaction. tuple[0] is the user's class, which can be
        1, 2 or 3. tuple[1] is a numpy array of 5 elemets, where every element i
        represents how many of the products i+1 the customer bought
    """

    # Competitor budget is added to the array of budgets
    budgets = np.insert(budgets, 0, env_data.competitor_budget)

    # Computing total number of customers for each class based on class_ratios
    customers_of_class_1 = int(num_customers * env_data.class_ratios[0])
    customers_of_class_2 = int(num_customers * env_data.class_ratios[1])
    customers_of_class_3 = num_customers - customers_of_class_1 - customers_of_class_2
    customers_per_class = [
        customers_of_class_1,
        customers_of_class_2,
        customers_of_class_3,
    ]

    total_interactions = list()

    # For every class, product ratios are computed
    for i in range(3):
        click_ratios = alpha_function(
            budgets,
            env_data.classes_parameters[i].steepness,
            env_data.classes_parameters[i].shift,
            env_data.classes_parameters[i].upper_bound,
        )

        # Replace ratios that are 0 with machine-espilon (10^-16) to ensure
        # compatibility with the Dirichlet function
        click_ratios = np.where(click_ratios == 0, 1e-16, click_ratios)

        alpha_ratios = rng.dirichlet(click_ratios)

        # This array will contain how many customers will land on a certain
        # product page, excluding the competitor page, which corresponds to the
        # first ratio
        product_ratios = [
            int(alpha_ratios[1] * customers_per_class[i]),
            int(alpha_ratios[2] * customers_per_class[i]),
            int(alpha_ratios[3] * customers_per_class[i]),
            int(alpha_ratios[4] * customers_per_class[i]),
            int(alpha_ratios[5] * customers_per_class[i]),
        ]

        # According to product ratios, for every product the computed number on
        # users are landed on the right and the interaction starts
        for product, ratio in enumerate(product_ratios):
            for interaction in range(ratio):
                user_class, items = _get_interaction(rng, i, product, env_data)
                total_interactions.append(Interaction(user_class, items))

    # Shuffle the list to make data more realistic
    rng.shuffle(total_interactions)
    return total_interactions


def _go_to_page(rng, user_class, primary_product, items_bought, env_data):

    """Shows the user a page of the specified primary products.

    After buying the primary product two secondary products are shown. Clicking
    on them shows the user another page where the clicked product is primary.

    Arguments:
        rng: instance of a generator (default_rng())

        user_class: integer from 1 to 3 representing user's class

        primary_product: integer representing the primary product. It goes from
            0 to 4 instead of 1 to 5 (so product-1 is 0 and so on)

        items_bought: numpy array of integers where every element i represents
            the quantity bought of the product i+1

        env_data: instance of Environment_data

    Returns:
        numpy array of integers where every element i represents
            the quantity bought of the product i+1
    """

    # Checks if the price of the primary product is under the reservation price of the user class
    if (
        env_data.product_prices[primary_product]
        < env_data.classes_parameters[user_class].reservation_price
    ):

        # The customer buys a random quantity of the product between 1 and max_tems
        items_bought[primary_product] += rng.integers(
            1, env_data.max_items, endpoint=True
        )

        secondary_products = env_data.next_products[primary_product]

        # If the user watches the first slot and clicks on the product he gets
        # redirected to a new primary product page where the secondary product
        # is the primary product
        if (
            rng.uniform(low=0.0, high=1.0)
            < env_data.graph[primary_product, secondary_products[0]]
            and not items_bought[secondary_products[0]]
        ):
            # Items bought on the opened page are added to the ones bought in
            # the current page
            items_bought = _go_to_page(
                rng, user_class, secondary_products[0], items_bought, env_data
            )

        # Same as before, if the user watches the second slot and clicks on it
        # he gets redirected to a new page where teh clicked product is thge
        # primary_product
        if (
            rng.uniform(low=0.0, high=1.0)
            < env_data.graph[primary_product, secondary_products[1]] * env_data.lam
            and not items_bought[secondary_products[1]]
        ):
            items_bought = _go_to_page(
                rng, user_class, secondary_products[1], items_bought, env_data
            )

    return items_bought
