from abc import ABC, abstractmethod


class Solver(ABC):

    @abstractmethod
    def __init__(self):
        self.horizon = None

    @abstractmethod
    def prepare(metacls):
        pass

    @abstractmethod
    def decide(self, observation_data, current_time):
        pass

    @abstractmethod
    def get_statistics(self):
        pass
