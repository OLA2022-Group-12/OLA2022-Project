from ola2022_project.learners.learner import Learner
import numpy as np

class StupidLearner(Learner):

    '''Learner that always subdivides equally the budget between products and doesn't
    update its prediction
    '''

    def __init__(self, total_budget, collected_rewards):
        super().__init__(total_budget, collected_rewards)

    def learn(self, reward):
        self.collected_rewards = np.append(self.collected_rewards, reward)
        
    def predict(self, data):
        return np.full((1, len(data.product_prices)), self.total_budget / len(data.product_prices))
