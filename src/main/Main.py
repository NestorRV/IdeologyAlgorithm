import optproblems
import numpy as np

from src.ia.IA import IdeologyAlgorithm


def main():
    dimension = 30
    n_parties = 5
    politicians = 150
    R = 0.05
    max_evaluations = 10000 * dimension
    desertion_threshold = 10

    f = optproblems.cec2005.F1(dimension)
    algorithm = IdeologyAlgorithm(n_parties=n_parties, politicians=politicians, R=R, function=f, function_index=1,
                                  max_evaluations=max_evaluations, desertion_threshold=desertion_threshold)
    best = algorithm.ideology_algorithm()
    print("F1 {} {} {}".format(best.get_fitness(), f.bias,
                               np.round(np.absolute((best.get_fitness() - f.bias) / f.bias * 100), 4)))

    f = optproblems.cec2005.F2(dimension)
    algorithm = IdeologyAlgorithm(n_parties=n_parties, politicians=politicians, R=R, function=f, function_index=2,
                                  max_evaluations=max_evaluations, desertion_threshold=desertion_threshold)
    best = algorithm.ideology_algorithm()
    print("F2 {} {} {}".format(best.get_fitness(), f.bias,
                               np.round(np.absolute((best.get_fitness() - f.bias) / f.bias * 100), 4)))

    f = optproblems.cec2005.F5(dimension)
    algorithm = IdeologyAlgorithm(n_parties=n_parties, politicians=politicians, R=R, function=f, function_index=5,
                                  max_evaluations=max_evaluations, desertion_threshold=desertion_threshold)
    best = algorithm.ideology_algorithm()
    print("F5 {} {} {}".format(best.get_fitness(), f.bias,
                               np.round(np.absolute((best.get_fitness() - f.bias) / f.bias * 100), 4)))

    f = optproblems.cec2005.F6(dimension)
    algorithm = IdeologyAlgorithm(n_parties=n_parties, politicians=politicians, R=R, function=f, function_index=6,
                                  max_evaluations=max_evaluations, desertion_threshold=desertion_threshold)
    best = algorithm.ideology_algorithm()
    print("F6 {} {} {}".format(best.get_fitness(), f.bias,
                               np.round(np.absolute((best.get_fitness() - f.bias) / f.bias * 100), 4)))

    f = optproblems.cec2005.F8(dimension)
    algorithm = IdeologyAlgorithm(n_parties=n_parties, politicians=politicians, R=R, function=f, function_index=8,
                                  max_evaluations=max_evaluations, desertion_threshold=desertion_threshold)
    best = algorithm.ideology_algorithm()
    print("F8 {} {} {}".format(best.get_fitness(), f.bias,
                               np.round(np.absolute((best.get_fitness() - f.bias) / f.bias * 100), 4)))

    f = optproblems.cec2005.F9(dimension)
    algorithm = IdeologyAlgorithm(n_parties=n_parties, politicians=politicians, R=R, function=f, function_index=9,
                                  max_evaluations=max_evaluations, desertion_threshold=desertion_threshold)
    best = algorithm.ideology_algorithm()
    print("F9 {} {} {}".format(best.get_fitness(), f.bias,
                               np.round(np.absolute((best.get_fitness() - f.bias) / f.bias * 100), 4)))

    f = optproblems.cec2005.F10(dimension)
    algorithm = IdeologyAlgorithm(n_parties=n_parties, politicians=politicians, R=R, function=f, function_index=10,
                                  max_evaluations=max_evaluations, desertion_threshold=desertion_threshold)
    best = algorithm.ideology_algorithm()
    print("F10 {} {} {}".format(best.get_fitness(), f.bias,
                                np.round(np.absolute((best.get_fitness() - f.bias) / f.bias * 100), 4)))

    f = optproblems.cec2005.F11(dimension)
    algorithm = IdeologyAlgorithm(n_parties=n_parties, politicians=politicians, R=R, function=f, function_index=11,
                                  max_evaluations=max_evaluations, desertion_threshold=desertion_threshold)
    best = algorithm.ideology_algorithm()
    print("F11 {} {} {}".format(best.get_fitness(), f.bias,
                                np.round(np.absolute((best.get_fitness() - f.bias) / f.bias * 100), 4)))

    f = optproblems.cec2005.F13(dimension)
    algorithm = IdeologyAlgorithm(n_parties=n_parties, politicians=politicians, R=R, function=f, function_index=13,
                                  max_evaluations=max_evaluations, desertion_threshold=desertion_threshold)
    best = algorithm.ideology_algorithm()
    print("F13 {} {} {}".format(best.get_fitness(), f.bias,
                                np.round(np.absolute((best.get_fitness() - f.bias) / f.bias * 100), 4)))

    f = optproblems.cec2005.F14(dimension)
    algorithm = IdeologyAlgorithm(n_parties=n_parties, politicians=politicians, R=R, function=f, function_index=14,
                                  max_evaluations=max_evaluations, desertion_threshold=desertion_threshold)
    best = algorithm.ideology_algorithm()
    print("F14 {} {} {}".format(best.get_fitness(), f.bias,
                                np.round(np.absolute((best.get_fitness() - f.bias) / f.bias * 100), 4)))

    f = optproblems.cec2005.F17(dimension)
    algorithm = IdeologyAlgorithm(n_parties=n_parties, politicians=politicians, R=R, function=f, function_index=17,
                                  max_evaluations=max_evaluations, desertion_threshold=desertion_threshold)
    best = algorithm.ideology_algorithm()
    print("F17 {} {} {}".format(best.get_fitness(), f.bias,
                                np.round(np.absolute((best.get_fitness() - f.bias) / f.bias * 100), 4)))

    f = optproblems.cec2005.F22(dimension)
    algorithm = IdeologyAlgorithm(n_parties=n_parties, politicians=politicians, R=R, function=f, function_index=22,
                                  max_evaluations=max_evaluations, desertion_threshold=desertion_threshold)
    best = algorithm.ideology_algorithm()
    print("F22 {} {} {}".format(best.get_fitness(), f.bias,
                                np.round(np.absolute((best.get_fitness() - f.bias) / f.bias * 100), 4)))

    f = optproblems.cec2005.F24(dimension)
    algorithm = IdeologyAlgorithm(n_parties=n_parties, politicians=politicians, R=R, function=f, function_index=24,
                                  max_evaluations=max_evaluations, desertion_threshold=desertion_threshold)
    best = algorithm.ideology_algorithm()
    print("F24 {} {} {}".format(best.get_fitness(), f.bias,
                                np.round(np.absolute((best.get_fitness() - f.bias) / f.bias * 100), 4)))


if __name__ == "__main__":
    main()
