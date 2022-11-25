import logging
import numpy as np


logger = logging.getLogger(__name__)


def budget_assignment(c):
    """Solve the budget assignment problem over the value matrix c.

    The matrix c is assumed to be divided into n rows (campaigns) and m columns
    (budget fractions). As the columns describe budget fractions from 0 to B,
    where b is the max budget, the function only needs to consider that the sum
    of the returned allocations/selections cannot exceed m - 1. To put it
    another way, column j repreresents the value of allocating j * (b / m) of
    the budget to the campaigns.

    Arguments:
        c: numpy matrix (nxm) where the value is the reward of assigning budget
           j to campaign i

    Returns:
        a numpy vector where row i has the index of selected column j of C
    """

    n, m = np.shape(c)
    if m == 0 or n == 0:
        return np.array([])

    # Setup a dp table (which will contain the computed cumulative value of a
    # certain allocation) and a table to store the allocations themselves. We
    # initialize both with the first input, as when there is nothing before us
    # we will always allocate the most possible to the single campaign.
    dp = [c[0]]
    allocs = [np.arange(m)]

    for i in range(1, n):
        next_dp_row = []
        next_alloc_row = []
        for j in range(m):
            # We reverse the previous row of the dynamic programming storage, to
            # easily compare the options we have by summing across the vectors
            prev_row_r = dp[i - 1][j::-1]
            curr_row = c[i][: j + 1]

            row_sum = prev_row_r + curr_row

            # The best index is the index which we should select for the maximum
            # profit, it will be what we store in allocation, and we will also
            # store the value of the row_sum at best_index to dp[i][j]
            best_index = np.argmax(row_sum)

            next_alloc_row.append(best_index)
            next_dp_row.append(row_sum[best_index])

        allocs.append(np.array(next_alloc_row))
        dp.append(np.array(next_dp_row))

    dp = np.array(dp)
    allocs = np.array(allocs)

    # We get the best allocation from the last row of the dp table
    best_dp_index = np.argmax(dp[-1])

    # As we stored the allocation of what the dp would mean, we now need to
    # trace back through the allocs array to create the final allocation
    last_alloc = allocs[-1][best_dp_index]

    # A list of the final allocations that will give the optimal budget
    # utilization
    final_allocs = [last_alloc]
    remaining_budget = m - last_alloc

    # This will loop backwards through allocs, excluding the last row, so from n - 2 to 0
    for i in range(n - 2, -1, -1):
        # Based on the remaining_budget, we take the max of the remaining
        # possible cumulative values that we have in the dynamic programming
        # table
        sub_dp = dp[i, :remaining_budget]
        sub_best_index = np.argmax(sub_dp) if len(sub_dp) > 0 else 0

        # Convert the index we chose into the actual allocation that was done
        # for the given sub_dp index
        next_alloc = allocs[i][sub_best_index]
        final_allocs.append(next_alloc)

        # As we might have taken more off the budget, we need to adjust it
        remaining_budget = remaining_budget - next_alloc

    # As we constructed the the final allocation from the back to front, we need
    # to reverse it before returning it
    return np.array(list(reversed(final_allocs)))
