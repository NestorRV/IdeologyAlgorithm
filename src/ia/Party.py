import numpy as np


class Party(object):
    def __init__(self, id):
        self.__id = id
        self.__politicians = None

    def get_id(self):
        return self.__id

    def get_politicians(self):
        return self.__politicians

    def get_leader(self):
        return self.__politicians[0]

    def get_subleader(self):
        return self.__politicians[1]

    def remove_last(self):
        self.__politicians = np.delete(self.__politicians, np.size(self.__politicians) - 1)

    def add_politician(self, p):
        self.__politicians = np.append(self.__politicians, p)

    def set_politicians(self, politicians):
        self.__politicians = politicians

    def sort_party(self):
        # Sort the politicians from min fitness to max (we are trying to minimize)
        self.__politicians = np.array(sorted(self.__politicians, key=lambda p: p.get_fitness()))
