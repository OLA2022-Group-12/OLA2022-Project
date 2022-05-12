import numpy as np

np.random.seed(1337)

class data:

    '''
    These are all dummy values. There is no particular criteria in how they were chosen.
    They can all be tuned to make the environment behave differently
    '''

    # Probability of every class to show up. They must add up to 1
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

    # Products graph's matrix. It's a dummy matrix, should be initialized with populate_graphs
    graph = np.random.rand(5,5)