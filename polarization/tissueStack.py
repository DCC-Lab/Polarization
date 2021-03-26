from .tissueLayer import TissueLayer
import numpy as np


class TissueStack:
    def __init__(self):
        self.layers = []

    def append(self, layer: TissueLayer):
        self.layers.append(layer)
