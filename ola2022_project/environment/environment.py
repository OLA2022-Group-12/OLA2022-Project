import numpy as np

'''
These are all dummy values. There is no particular criteria in how they were chosen.
They can all be tuned to make the environment behave differently
'''

# Probability of every class to show up. They should add up to 1
class_ratios = [0.3, 0.6, 0.1]

# Reservation price of every class
reservation_prices = [10, 30, 20]

# Price of the 5 products
product_prices = [10, 15, 25, 18, 5]

# Parameters of the alpha functions for every class. The functions sigmoidal.
# The first value is the steepness, the second is the shift and the third is the
# upper bound.
parameters = [[0.17,4.5,50], [0.15,5,65], [0.22,5.3,100]]

# The competitor budget is assumed to be constant, since the competitor is
# non-strategic
competitor_budget = 100

# Lambda parameter, which is the probability of osserving the next secondary product
# according to the project's assignment
lam = 0.5

# Max number of items a customer can buy of a certain product. The number of
# bought items is determined randomly with max_items as upper bound
max_items = 3

np.random.seed(1337)

# Products graph's matrix. It's a dummy matrix, should be initialized with populate_graphs
graph = np.random.rand(5,5)


def alpha_function(budget, steepness, shift, upper_bound):

    '''Alpha function with the shape of a sigmoidal function. Computes the
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
    '''

    return upper_bound * (1 / ( 1 + np.exp(-steepness*budget + shift) ))


def populate_graph(size=5, fully_connected=True, zeros_probability=0.5):

    ''' This function should always be called before using the environment.
    By default generates a 5x5 fully connected graph. The graph has no
    auto-loops

    Arguments:
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
    '''

    # Generates a random fully-connected graph
    graph = np.random.rand(size,size)

    # Removes some connections if graph is requested to be not fully-connected
    if not fully_connected:
        mask = np.random.choice([True, False], (size,size), p=[zeros_probability, 1-zeros_probability])
        graph[mask] = 0

    # Removes auto-loops from graph
    np.fill_diagonal(graph, 0)


def get_interaction(user_class, primary_product):
    '''Computes a single interaction and returns it.

    This method shouldn't be called on its own, but only via the
    get_day_of_interactions() method

    Arguments:
        user_class: integer from 1 to 3 representing user's class
        primary_product: integer from 0 to 5 representing product number. 
            Note: product 0 is the competitor's

    Returns:
        A tuple, the first element is an integer (1, 2 or 3) representing the
        user's class of the interaction, the second is an numpy array of 5
        elements, where every element i is an integer indicating the quantity
        bought of the product i+1    
    '''

    # This array is initialized with zeros and represents the quantity of bought
    # items for every product
    items_bought = np.zeros(5)

    # If primary_product != 0 it means that the user landed on a page of one of
    # the 5 products, otherwise the user landed on a competitor product's page
    if primary_product != 0:
        primary_product -= 1
        go_to_page(user_class, primary_product, items_bought)

    return user_class, items_bought


def get_day_of_interactions(num_customers, budgets):
    
    ''' Main method to be called when interacting with the environment. Outputs
    all the interactions of an entire day. When called generates new alphas from
    the clicks function with budget as input.

    Arguments:
        num_customers: total number of customers that will make interactions in
            a day
        
        budgets: numpy array of 5 integers containing the budget for each
            product. The i-element is the budget for the i+1 product
    
    Returns:
        A list of tuples, with a size corresponding to num_customers. Every
        tuple denotes an interaction. tuple[0] is the user's class, which can be
        1, 2 or 3. tuple[1] is a numpy array of 5 elemets, where every element i
        represents how many of the products i+1 the customer bought
    '''

    # Competitor budget is added to the array of budgets
    budgets = np.insert(budgets, 0, competitor_budget)

    # Computing total number of customers for each class based on class_ratios
    customers_of_class_1 = int(num_customers*class_ratios[0])
    customers_of_class_2 = int(num_customers*class_ratios[1])
    customers_of_class_3 = num_customers - customers_of_class_1 - customers_of_class_2
    customers_per_class = [customers_of_class_1, customers_of_class_2, customers_of_class_3]

    total_interactions = list()

    # For every class, product ratios are computed
    for i in range(3):
        click_ratios = alpha_function(budgets, parameters[i][0], parameters[i][1], parameters[i][2])
        alpha_ratios = np.random.dirichlet(click_ratios)
        
        # This array will contain how many customers will land on a certain product page
        product_ratios = [int(alpha_ratios[0]*customers_per_class[i]),
             int(alpha_ratios[1]*customers_per_class[i]),
             int(alpha_ratios[2]*customers_per_class[i]),
             int(alpha_ratios[3]*customers_per_class[i]),
             int(alpha_ratios[4]*customers_per_class[i])]
        product_ratios.append(customers_per_class[i] - sum(product_ratios))

        # According to product ratios, for every product the computed number on
        # users are landed on the right and the interaction starts
        for idx, ratio in enumerate(product_ratios):
            for interaction in range(ratio):
                user_class, items = get_interaction(i, idx)
                total_interactions.append((user_class, items))

    # Shuffle the list to make data more realistic
    np.random.shuffle(total_interactions)
    return total_interactions


def go_to_page(user_class, primary_product, items_bought):

    '''Shouldn't be called from outside. Shows the user a page of the specified
    primary products.
    
    After buying the primary product two secondary products are shown. Clicking
    on them shows the user another page where the clicked product is primary.
    
    Arguments:
        primary_product: integer representing the primary product. It goes from
            0 to 4 instead of 1 to 5 (so product-1 is 0 and so on)
        items_bought: numpy array of integers where every element i represents
            the quantity bought of the product i+1
    '''
    
    # Checks if the price of the primary product is under the reservation price of the user class
    if product_prices[primary_product] < reservation_prices[user_class]:
        
        # The customer buys a random quantity of the product between 1 and max_tems
        items_bought[primary_product] += np.random.randint(1, max_items)

        # Chooses 2 secondary products from all the 5 products excluding the current primay product
        available_prod = list(range(5))
        available_prod.remove(primary_product)
        secondary_products = np.random.choice(available_prod, 2, replace=False)

        # If the user watches the second slot and clicks on the product he gets
        # redirected to a new primary product page where the secondary product
        # is the primary product
        if np.random.rand() < graph[primary_product, secondary_products[0]] and not items_bought[secondary_products[0]]:
            go_to_page(user_class, secondary_products[0], items_bought)
        
        # When the user does not click on the secondary product in the first
        # slot if he observes and click on the second slot he gets redirected to
        # a new page where this secondary product is the primary product
        elif np.random.rand() < graph[primary_product, secondary_products[1]] * lam and not items_bought[secondary_products[1]]:
            go_to_page(user_class, secondary_products[1], items_bought)