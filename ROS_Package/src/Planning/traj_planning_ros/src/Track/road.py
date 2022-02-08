import numpy as np
from abc import ABC, abstractmethod

class Section(ABC):
    @abstractmethod
    def route_to_global(self):
        pass

    def plot(self):
        pass

class Road(ABC):
    @abstractmethod

    def add_straight(self):
        pass

    def add_curve(self):
        pass
    
    def reference_traj(self):
        pass

    def plot(self):
        pass