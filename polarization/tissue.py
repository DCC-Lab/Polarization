from .tissueStack import TissueStack
import numpy as np


class Tissue:
    def __init__(self, shape):
        self.map = np.zeros(shape)
        self.layers = []

    def stackAt(self, coordinates) -> TissueStack:
        pass

    def display(self):
        """ Display all layer stacks and their properties. """
        pass


class RandomTissue2D(Tissue):
    def __init__(self, lines=100, height=3000):
        pass
