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
        weights = [1 / ((k + 1) ** s) for k in range(elements_list) ]
        total = sum(weights)
        return [w / total for w in weights]

class System:
    """
    Class that models the system and client requests.


    """
    def __init__(self, states_list, s = 1):
        self.states_list = states_list
        self.states_prob = ProbGenerator.gen_zipf_prob(states_list, s)

    def __str__(self) -> str:
        return f"Conjunto de estados = {self.states_list}, D.P = {self.states_prob}"


def main():
    help(ProbGenerator.gen_zipf_prob)


if __name__ == "__main__":
    main()

