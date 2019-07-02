import copy
import sys

import numpy as np

from src.ia.Party import Party
from src.ia.Politician import Politician
from src.util.constrained_kmeans import constrained_kmeans


class IdeologyAlgorithm:
    def __init__(self, n_parties, politicians, R, function, function_index, max_evaluations, desertion_threshold):
        if (politicians % n_parties != 0):
            print("It's impossible to create", n_parties, "parties with the same amount of politicians in each one.")
            raise SystemExit

        np.random.seed(3)

        self.__n_parties = n_parties
        self.__politicians = politicians
        self.__R = R
        self.__function = function
        self.__function_index = function_index
        self.__evaluations = 0
        self.__max_evaluations = max_evaluations
        self.__desertion_threshold = desertion_threshold

        self.__best_value_file = open(
            "./results/best_value_" + str(function_index) + "_dim" + str(self.__function.num_variables) + ".dat", "w")

        self.__best_solution = Politician(self.__function, self.__function_index)
        # The maximum available value
        self.__best_solution.set_fitness(sys.float_info.max)
        self.__population = None

        if self.__function_index == 7:
            self.__original_lower_bounds = np.array([0 for i in range(self.__function.num_variables)])
            self.__original_upper_bounds = np.array([600 for i in range(self.__function.num_variables)])
        elif self.__function_index == 25:
            self.__original_lower_bounds = np.array([-2 for i in range(self.__function.num_variables)])
            self.__original_upper_bounds = np.array([5 for i in range(self.__function.num_variables)])
        else:
            self.__original_lower_bounds = np.array([x for x in self.__function.min_bounds])
            self.__original_upper_bounds = np.array([x for x in self.__function.max_bounds])

    def __initialize_population(self):
        # Random politicians
        population = np.array([Politician(self.__function, self.__function_index) for i in range(self.__politicians)])
        # Make the parties using k-means
        (centroids, assignment, f) = constrained_kmeans([p.get_solution() for p in population],
                                                        np.repeat(self.__politicians / self.__n_parties,
                                                                  self.__n_parties))
        parties = [list() for i in range(self.__n_parties)]
        # Repart the politicians
        [parties[i].append(p) for p, i in zip(population, assignment)]

        # Save the population
        self.__population = np.array([Party(i) for i in range(self.__n_parties)])
        [p.set_politicians(np.array(party)) for p, party in zip(self.__population, parties)]

    def __sort_population(self):
        # Sort the parties
        [party.sort_party() for party in self.__population]

    def __update_leader(self, leader, id_party, global_leader):
        lower = np.array([p.get_lower_bounds() for p in self.__population[id_party].get_politicians()])
        upper = np.array([p.get_upper_bounds() for p in self.__population[id_party].get_politicians()])
        lower_bounds = np.min(lower, axis=0)
        upper_bounds = np.max(upper, axis=0)

        # The leader's update has three parts: introspection, local competition and global competition
        insp_lower_bounds = np.array([i - self.__R * np.absolute(u - l) for i, l, u in
                                      zip(leader.get_solution(), lower_bounds, upper_bounds)])

        insp_upper_bounds = np.array([i + self.__R * np.absolute(u - l) for i, l, u in
                                      zip(leader.get_solution(), lower_bounds, upper_bounds)])

        local_lower_bounds = np.array([i - self.__R * np.absolute(u - l) for i, l, u in
                                       zip(self.__population[id_party].get_subleader().get_solution(),
                                           lower_bounds, upper_bounds)])

        local_upper_bounds = np.array([i + self.__R * np.absolute(u - l) for i, l, u in
                                       zip(self.__population[id_party].get_subleader().get_solution(),
                                           lower_bounds, upper_bounds)])

        global_lower_bounds = np.array([i - self.__R * np.absolute(u - l) for i, l, u in
                                        zip(global_leader.get_solution(), lower_bounds,
                                            upper_bounds)])

        global_upper_bounds = np.array([i + self.__R * np.absolute(u - l) for i, l, u in
                                        zip(global_leader.get_solution(), lower_bounds,
                                            upper_bounds)])

        # We need to truncate the bounds if they are out of the accepted ones
        for l, u, i in zip(self.__original_lower_bounds, self.__original_upper_bounds,
                           range(np.size(self.__original_lower_bounds))):
            if insp_lower_bounds[i] < l or insp_lower_bounds[i] > u:
                insp_lower_bounds[i] = l
            if insp_upper_bounds[i] > u or insp_upper_bounds[i] < l or insp_upper_bounds[i] < insp_lower_bounds[i]:
                insp_upper_bounds[i] = u

            if local_lower_bounds[i] < l or local_lower_bounds[i] > u:
                local_lower_bounds[i] = l
            if local_upper_bounds[i] > u or local_upper_bounds[i] < l or local_upper_bounds[i] < local_lower_bounds[i]:
                local_upper_bounds[i] = u

            if global_lower_bounds[i] < l or global_lower_bounds[i] > u:
                global_lower_bounds[i] = l
            if global_upper_bounds[i] > u or global_upper_bounds[i] < l or global_upper_bounds[i] < \
                    global_lower_bounds[i]:
                global_upper_bounds[i] = u

        insp_solution = np.array([np.random.uniform(x, y) for x, y in zip(insp_lower_bounds, insp_upper_bounds)])
        local_solution = np.array([np.random.uniform(x, y) for x, y in zip(local_lower_bounds, local_upper_bounds)])
        global_solution = np.array([np.random.uniform(x, y) for x, y in zip(global_lower_bounds, global_upper_bounds)])

        insp_fitness = self.__function(insp_solution)
        self.__evaluations += 1
        local_fitness = self.__function(local_solution)
        self.__evaluations += 1
        global_fitness = self.__function(global_solution)
        self.__evaluations += 1

        results = np.array([insp_fitness, local_fitness, global_fitness])
        # We need to transform the results to manage negative values and to
        # transform the minimization problem in a maximization one
        transformed = np.max(results) - (results - np.min(results))
        # Get the ranges of probabilities for roulette wheel selection
        probability_1 = 0 + transformed[0] / np.sum(transformed)
        probability_2 = probability_1 + transformed[1] / np.sum(transformed)

        #  Apply roulette wheel selection
        # The leader is updated according to the selected individual
        r = np.random.uniform(0, 1)
        if 0 < r <= probability_1:
            leader.set_solution(insp_solution)
            leader.set_fitness(insp_fitness)
            leader.set_lower_bounds(insp_lower_bounds)
            leader.set_upper_bounds(insp_upper_bounds)
        elif probability_1 < r <= probability_2:
            leader.set_solution(local_solution)
            leader.set_fitness(local_fitness)
            leader.set_lower_bounds(local_lower_bounds)
            leader.set_upper_bounds(local_upper_bounds)
        elif probability_2 < r <= 1:
            leader.set_solution(global_solution)
            leader.set_fitness(global_fitness)
            leader.set_lower_bounds(global_lower_bounds)
            leader.set_upper_bounds(global_upper_bounds)

    def __update_party(self, party):
        lower = np.array([p.get_lower_bounds() for p in party.get_politicians()])
        upper = np.array([p.get_upper_bounds() for p in party.get_politicians()])
        lower_bounds = np.min(lower, axis=0)
        upper_bounds = np.max(upper, axis=0)

        for individual, index in zip(party.get_politicians(), range(np.size(party.get_politicians()))):
            if index != 0 and index != np.size(party.get_politicians()) - 1:
                # The individual's update has two parts: introspection and local competition
                insp_lower_bounds = np.array([i - self.__R * np.absolute(u - l) for i, l, u in
                                              zip(individual.get_solution(), lower_bounds,
                                                  upper_bounds)])

                insp_upper_bounds = np.array([i + self.__R * np.absolute(u - l) for i, l, u in
                                              zip(individual.get_solution(), lower_bounds,
                                                  upper_bounds)])

                local_lower_bounds = np.array([i - self.__R * np.absolute(u - l) for i, l, u in
                                               zip(party.get_leader().get_solution(),
                                                   lower_bounds, upper_bounds)])

                local_upper_bounds = np.array([i + self.__R * np.absolute(u - l) for i, l, u in
                                               zip(party.get_leader().get_solution(),
                                                   lower_bounds, upper_bounds)])

                # We need to truncate the bounds if they are out of the accepted ones
                for l, u, i in zip(self.__original_lower_bounds, self.__original_upper_bounds,
                                   range(np.size(lower_bounds))):
                    if insp_lower_bounds[i] < l or insp_lower_bounds[i] > u:
                        insp_lower_bounds[i] = l
                    if insp_upper_bounds[i] > u or insp_upper_bounds[i] < l:
                        insp_upper_bounds[i] = u

                    if local_lower_bounds[i] < l or local_lower_bounds[i] > u:
                        local_lower_bounds[i] = l
                    if local_upper_bounds[i] > u or local_upper_bounds[i] < l:
                        local_upper_bounds[i] = u

                insp_solution = np.array(
                    [np.random.uniform(x, y) for x, y in zip(insp_lower_bounds, insp_upper_bounds)])
                local_solution = np.array(
                    [np.random.uniform(x, y) for x, y in zip(local_lower_bounds, local_upper_bounds)])

                insp_fitness = self.__function(insp_solution)
                self.__evaluations += 1
                local_fitness = self.__function(local_solution)
                self.__evaluations += 1

                results = np.array([insp_fitness, local_fitness])
                # We need to transform the results to manage negative values and to
                # transform the minimization problem in a maximization one
                transformed = np.max(results) - (results - np.min(results))
                # Get the ranges of probabilities for roulette wheel selection
                probability_1 = 0 + transformed[0] / np.sum(transformed)

                #  Apply roulette wheel selection
                # The leader is updated according to the selected individual
                r = np.random.uniform(0, 1)
                if 0 < r <= probability_1:
                    individual.set_solution(insp_solution)
                    individual.set_fitness(insp_fitness)
                    individual.set_lower_bounds(insp_lower_bounds)
                    individual.set_upper_bounds(insp_upper_bounds)
                elif probability_1 < r <= 1:
                    individual.set_solution(local_solution)
                    individual.set_fitness(local_fitness)
                    individual.set_lower_bounds(local_lower_bounds)
                    individual.set_upper_bounds(local_upper_bounds)

    def ideology_algorithm(self):
        self.__initialize_population()
        self.__sort_population()

        while self.__evaluations < self.__max_evaluations:
            # Update the leader of each party
            leaders = np.array([party.get_leader() for party in self.__population])
            global_leader = copy.deepcopy(sorted(leaders, key=lambda p: p.get_fitness())[0])
            self.__best_value_file.write("{} {} \n".format(self.__evaluations, global_leader.get_fitness()))

            # We must store the best solution in every iteration
            if (global_leader.get_fitness() < self.__best_solution.get_fitness()):
                self.__best_solution = copy.deepcopy(global_leader)

            [self.__update_leader(l, i, global_leader) for l, i in zip(leaders, range(self.__n_parties)) if l != None]

            # The worst individual may desert
            for party, id in zip(self.__population, range(self.__n_parties)):
                if np.size(party.get_politicians()) > 2:
                    if (np.absolute(party.get_politicians()[-1].get_fitness()) - np.absolute(
                            party.get_politicians()[-2].get_fitness())) > self.__desertion_threshold:
                        # We store the index of the parties
                        index = np.array(range(self.__n_parties))
                        # and delete the one we are processing
                        index = np.delete(index, id)
                        # We shuffle the index to choose the new party for the desertor
                        np.random.shuffle(index)
                        deserter = copy.deepcopy(party.get_politicians()[-1])
                        self.__population[id].remove_last()
                        self.__population[index[0]].add_politician(deserter)

            # The deserter may be a good one, so...
            self.__sort_population()
            # We update all the individuals except the leader and the worst
            [self.__update_party(party) for party in self.__population]
            # We sort the population to get the best individuals first
            self.__sort_population()

        self.__best_value_file.close()

        leaders = np.array([party.get_leader() for party in self.__population])
        if sorted(leaders, key=lambda p: p.get_fitness())[0].get_fitness() < self.__best_solution.get_fitness():
            return sorted(leaders, key=lambda p: p.get_fitness())[0]
        else:
            return self.__best_solution

    def get_population(self):
        return self.__population
