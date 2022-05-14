from abc import ABC, abstractmethod
import numpy as np

class Learner(ABC):

    '''Generic Learner interface for interactive agents capable of learning the budget 
    distribution for a set of subcampaigns from the environment they exist in and the 
    online feedback returned.
    '''

    def __init__(self, total_budget, collected_rewards):
        self._total_budget = total_budget
        self._collected_rewards = collected_rewards

    @property
    def total_budget(self):

        '''Overall total budget that the learner should subdivide optimally between
        products.
        '''

        return self._total_budget

    @total_budget.setter
    def total_budget(self, value):
        self._total_budget = value

    @property
    def collected_rewards(self):

        '''Array of values representing all of the rewards collected by the learner
        during its life
        '''

        return self._collected_rewards

    @collected_rewards.setter
    def collected_rewards(self, value):
        self._collected_rewards = value

    @classmethod
    @abstractmethod
    def learn(self, reward, prediction):

        '''Updates the learner's properties according to the reward received.

        Arguments: 
            reward: the reward obtained from the environment, needed for the tuning of 
                internal properties done by the learner
            prediction: array containing the previous budget evaluation of the learner
        '''

        pass

    @classmethod
    @abstractmethod
    def predict(self, data):

        '''Makes an inference about the values of the budgets for the subcampaigns
        utilizing the information gathered over time and the current state of the
        environment

        Arguments: 
            data: up-to-date, complete or incomplete environment information that is 
                used by the learner in order to make the inference

        Returns: 
            a list of values, corresponding to the budgets inferred given the knowledge
            obtained by the learner until now
        '''

        pass
