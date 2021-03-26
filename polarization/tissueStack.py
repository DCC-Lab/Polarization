from .tissueLayer import TissueLayer
import numpy as np


class TissueStack:
    def __init__(self, offset=0):
        self.layers = []
        self.offset = offset
        self.layerPositions = [self.offset]

    def append(self, layer: TissueLayer):
        self.layerPositions.append(self.layerPositions[-1] + layer.thickness)
        self.layers.append(layer)

    def __iter__(self):
        return self.layers.__iter__()
