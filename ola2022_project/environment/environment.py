import numpy as np

'''
These are all dummy values. There is no particular criteria in how they were chosen.
They can all be tuned to make the environment behave differently
'''

# Probability of every class to show up. They should add up to 1
class_probabilities = [0.3, 0.6, 0.1]

# Reservation price of every class
reservation_prices = [10, 30, 20]

# Price of the 5 products
product_prices = [10, 15, 25, 18, 5]

# Dirichlet alpha variables for every class
alpha_ratios = [[20, 10, 20, 20, 5, 30], [15, 25, 20, 5, 40, 15], [10, 20, 40, 5, 5, 20]]

# Lambda parameter, which is the probability of osserving the next secondary product
# according to the project's assignment
lam = 0.5

# Max number of items a customer can buy of a certain product. The number of
# bought items is determined randomly with max_items as upper bound
max_items = 3

np.random.seed(1337)

graph = np.random.rand(5,5)


def populate_graph(size=5, fully_connected=True, zeros_probability=0.5):

    ''' This function should always be called before using the environment.
    By default generates a 5x5 fully connected graph.

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

    


def get_interaction(alphas):
    '''Computes a single interaction and returns it.

    This method shouldn't be called on its own, but only via the
    get_day_of_interactions() method

    Arguments:
        alphas: list of 3 array of 6 elements representing the alpha ratios for
            all the three user classes

    Returns:
        A tuple, the first element is an integer (1, 2 or 3) representing the
        user's class of the interaction, the second is an numpy array of 5
        elements, where every element i is an integer indicating the quantity
        bought of the product i+1    
    '''

    # User's class is chosen randomly with weighted probability
    user_class = np.random.choice([1, 2, 3], p=class_probabilities)

    # This array is initialized with zeros and represents the quantity of bought
    # items for every product
    items_bought = np.zeros(5)

    primary_product = np.random.choice(list(range(6)), p=alphas[user_class])

    # If primary_product != 0 it means that the user landed on a page of one of
    # the 5 products, otherwise the user landed on a competitor product's page
    if primary_product != 0:
        primary_product -= 1
        go_to_page(primary_product, user_class, items_bought)

    return user_class, items_bought


def get_day_of_interactions(num_customers):
    
    ''' Main method to be called when interacting with the environment. Outputs
    all the interactions of an entire day. When called discards the old alpha
    values and generates new alphas from a new realization of the Dirichlet
    distribution. If called again it will be considered a new day.

    Arguments:
        num_customers: total number of customers that will make interactions in
            a day
    
    Returns:
        A list of tuples, with a size corresponding to num_customers. Every
        tuple denotes an interaction. tuple[0] is the user's class, which can be
        1, 2 or 3. tuple[1] is a numpy array of 5 elemets, where every element i
        represents how many of the products i+1 the customer bought
    '''

    # Alphas are initialized
    alphas = list()
    for i in range(3):
        alphas.append(np.random.dirichlet(alpha_ratios[i]))

    total_interactions = list()

    # num_customers interactions are performed, then saved and returned
    for i in range(num_customers):
        user_class, items = get_interaction(alphas)
        total_interactions.append((user_class, items))

    return total_interactions


def go_to_page(user_class, primary_product, items_bought):
    '''Shows the user a page of the specified primary products. 
    
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
            go_to_page(secondary_products[0], items_bought)
        
        # When the user does not click on the secondary product in the first
        # slot if he observes and click on the second slot he gets redirected to
        # a new page where this secondary product is the primary product
        elif np.random.rand() < graph[primary_product, secondary_products[1]] * lam and not items_bought[secondary_products[1]]:
            go_to_page(secondary_products[1], items_bought)