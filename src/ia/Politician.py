import numpy as np
import itertools


class Politician(object):
    id_generator = itertools.count(1)

    def __init__(self, f, index):
        if index == 7:
            self.__lower_bounds = np.array([0 for _ in range(f.num_variables)])
            self.__upper_bounds = np.array([600 for _ in range(f.num_variables)])
        elif index == 25:
            self.__lower_bounds = np.array([-2 for _ in range(f.num_variables)])
            self.__upper_bounds = np.array([5 for _ in range(f.num_variables)])
        else:
            self.__lower_bounds = np.array([x for x in f.min_bounds])
            self.__upper_bounds = np.array([x for x in f.max_bounds])

        self.__solution = np.array([np.random.uniform(x, y) for x, y in zip(self.__lower_bounds, self.__upper_bounds)])

        self.__fitness = f(self.__solution)
        self.__id = next(self.id_generator) - 1

    def get_id(self):
        return self.__id

    def get_solution(self):
        return self.__solution

    def set_solution(self, solution):
        self.__solution = solution

    def get_lower_bounds(self):
        return self.__lower_bounds

    def set_lower_bounds(self, bounds):
        self.__lower_bounds = bounds

    def get_upper_bounds(self):
        return self.__upper_bounds

    def set_upper_bounds(self, bounds):
        self.__upper_bounds = bounds

    def get_fitness(self):
        return self.__fitness

    def set_fitness(self, fitness):
        self.__fitness = fitness
