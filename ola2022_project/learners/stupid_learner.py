from learner import Learner
import numpy as np

class StupidLearner(Learner):

    def learn(self, reward):
        self.collected_rewards = np.append(self.collected_rewards, reward)
        
    def infer(self, data):
        return np.full((1, len(data.prices)), self.total_budget / len(data.prices))
