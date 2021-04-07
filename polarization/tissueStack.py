from .tissueLayer import *
from .jonesvector import JonesVector
from .jonesmatrix import JonesMatrix
from typing import List
import numpy as np


class TissueStack:
    def __init__(self, offset=0, layers=None):
        self.layers: List[TissueLayer] = []
        self.offset = offset

        for layer in layers:
            self.append(layer)

    def append(self, layer: TissueLayer):
        self.layers.append(layer)

    def __iter__(self):
        return iter(self.layers)

    def transferMatrix(self, layerIndex=None):
        # todo: this is missing the initial propagation in 'vacuum' with L=offset
        M = JonesMatrix(1, 0, 0, 1)  # * np.exp(1j * k * self.offset)
        for layer in self.layers[: layerIndex]:
            M *= layer.transferMatrix()
        return M

    def propagateThrough(self, vector: JonesVector) -> JonesVector:
        return self.transferMatrix() * vector

    def propagateManyThrough(self, vectors: List[JonesVector]) -> List[JonesVector]:
        J = []
        for v in vectors:
            J.append(self.propagateThrough(v))
        return J

    def backscatter(self, vector: JonesVector) -> JonesVector:
        signal = JonesVector(0, 0, k=vector.k)
        for i, layer in enumerate(self.layers):
            signal += self.transferMatrix(i).backward() * (self.transferMatrix(i) * layer.backscatter(vector))
        return signal

    def backscatterMany(self, vectors: List[JonesVector]) -> List[JonesVector]:
        vectorsOut = []
        for v in vectors:
            vectorsOut.append(self.backscatter(v))
        return vectorsOut


class RandomTissueStack(TissueStack):
    def __init__(self, surface=True, maxBirefringence=0.0042, nLayers=None, offset=None):
        if offset is None:
            offset = np.random.randint(200, 600)
        super(RandomTissueStack, self).__init__(offset=offset)

        if surface:
            self.append(SurfaceTissueLayer())
        if nLayers is None:
            nLayers = np.random.randint(1, 10)
        for layer in range(nLayers):
            self.append(RandomTissueLayer(maxBirefringence=maxBirefringence))
