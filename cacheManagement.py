import numpy as np
import random

class ProbGenerator:
    """
    Class with static methods for probabilities generation
    """

    @staticmethod
    def gen_zipf_prob(elements_list, s = 1):
        """
        Generates the D.P for the given elements list.

            Parameters:

                elements_list: 
                A list with the elements to assign probability. It should be 
                sorted from most frequent to less.

                s:
                The bias parameter of the distribution.

            Return value:

                List with the corresponding probabilities.
        """
        weights = [1 / ((k + 1) ** s) for k in range(len(elements_list)) ]
        total = sum(weights)
        return [w / total for w in weights]

class System:
    """
    Class that models the system and client requests.

        Attributes:

            objs_latency: 
                A list representing the objects set. It must be 
                sorted from most frequent to less. The index i of the list
                corresponds to the object (i, l).

            objs_prob:
                A list with the corresponding Zipf D.P for the objects.

            objs_number:
                Number of objects available.
    """
    def __init__(self, objs_list):
        self.objs_latency = [obj[1] for obj in objs_list]
        self.objs_prob = ProbGenerator.gen_zipf_prob(objs_list)
        self.objs_number = len(self.objs_latency)

    def __str__(self) -> str:
        return f"System\n    Latencias = {self.objs_latency},\n    D.P = {self.objs_prob}\n\n"

    def get_objs_number(self) -> int:
        return self.objs_number
    
    def get_obj_latency(self, id) -> int:
        """
        Returns the latency of object with identifier id.

        Parameters:

            id:
                Object identifier.
        """
        return self.objs_latency[id]

    def get_client_request(self):
        """
        Returns a random client request (object id) 
        attending to the objects D.P.
        """
        return random.choices(
                range(self.objs_number), 
                weights=self.objs_prob,
                k=1
        )[0]
    

class Agent:
    """
    Class that models the system cache memory management and learns by
    Q-learning.

        Attributes:

            system_reference: 
                A reference to a System instance.

            q_dict:
                A dictionary representing the Q-function estimates.

            discount_factor:
                The discount factor used to estimate the Q-function.

            cache_capacity:
                The max capacity of the cache memory. Must be less or equal than the
                number of objects available.

            current_state:
                Represents the current state of the cache memory.

            trials:
                Number of trials that have been done.

            explotarion_e:
                The epsilon parameter for e-greedy exploration.

            learning_rate:
                The learning rate for the Q-learning method.
    """
    def __init__(self, system_reference, discount_factor, cache_capacity):
        self.system_reference = system_reference
        self.q_dict = dict()
        self.discount_factor = discount_factor
        self.exploration_e = 1.0
        self.trials_number = 0
        self.cache_capacity = cache_capacity
        # As initial state it fills the cache memory randomly
        self.current_state = set(
            random.sample( 
                range(self.system_reference.get_objs_number()), 
                self.cache_capacity
            )
        )

        self.accum_reward = 0
        self.accum_hits = 0
        self.accum_latency = 0

        self.metric_accum_rewards = np.array([])
        self.metric_hit_rates = np.array([])
        self.metric_mean_latencies = np.array([])

    def __str__(self):
        return f"""Agent
    Q-dictionary: {self.q_dict},
    Discount factor (gamma): {self.discount_factor},
    Exploration epsilon: {self.exploration_e},
    Current state: {self.current_state},
    Trials: {self.trials_number},


    """

    def trial(self):
        requested_object = self.system_reference.get_client_request()

        if requested_object in self.current_state:
            reward = 0
            hit = 1
            latency = 0

        self.updateMetrics(reward, hit, latency)

    def updateMetrics(self, reward, hit, latency):
        """
        Updates metrics after every trial.

        Parameters:

            reward:
                Reward obtained.

            hit:
                0 for miss and 1 for hit.

            latency:
                Latency for retrieving the requested object. 
        """
        self.trials_number += 1

        self.accum_reward += reward
        self.accum_hit += hit
        self.accum_latency += latency
        
        self.metric_accum_rewards.append(self.accum_reward)
        self.metric_hit_rates.append(self.hits / self.trials_number)
        self.metric_mean_latencies.append(self.accum_latency / self.trials_number)


class Test:
    """
    Static methods for Testing.
    """
    @staticmethod
    def get_client_request(system, trials):
        frequencies = [0] * system.get_objs_number()

        for i in range(trials):
            frequencies[system.get_client_request()] += 1

        print(frequencies)


def main():
    random_seed = 33
    discount_factor = 0.9
    cache_capacity = 1

    random.seed(random_seed)

    states = [(1, 100), (2, 30), (3, 45)]
    system = System(states)

    agent = Agent(system, discount_factor, cache_capacity)

    print(system)
    print(agent)
    
    Test.get_client_request(system, 10000)


if __name__ == "__main__":
    main()

