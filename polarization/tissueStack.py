from .tissueLayer import *
from .jonesvector import JonesVector
from .jonesmatrix import JonesMatrix, Vacuum
from .pulse import Pulse
from typing import List
import numpy as np

__all__ = ['TissueStack', 'RandomTissueStack']


class TissueStack:
    def __init__(self, offset=0, layers=None):
        self.layers: List[TissueLayer] = []
        self.offset = offset

        if layers is not None:
            for layer in layers:
                self.append(layer)

    def append(self, layer: TissueLayer):
        self.layers.append(layer)

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def transferMatrix(self, layerIndex=None, backward=False):
        M = JonesMatrix(1, 0, 0, 1)
        if backward:
            backwardLayers = self.layers[:layerIndex]
            backwardLayers.reverse()
            for layer in backwardLayers:
                M *= layer.transferMatrix().backward()
            M *= Vacuum(physicalLength=self.offset)
        else:
            M *= Vacuum(physicalLength=self.offset)
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
            signal += self.transferMatrix(i, backward=True) * (self.transferMatrix(i) * layer.backscatter(vector))
        return signal

    def backscatterMany(self, vectors: List[JonesVector]):
        vectorsOut = []
        for v in vectors:
            vectorsOut.append(self.backscatter(v))

        if type(vectors) is Pulse:
            return Pulse(vectors=vectorsOut)
        else:
            return vectorsOut


class RandomTissueStack(TissueStack):
    def __init__(self, surface=True, maxBirefringence=0.0042, nLayers=None, offset=None, layerHeightRange=(60, 400)):
        self.params = {'surface': surface, 'maxBirefringence': maxBirefringence, 'nLayers': nLayers,
                       'offset': offset, 'layerHeightRange': layerHeightRange}

        if offset is None:
            offset = np.random.randint(200, 600)
        super(RandomTissueStack, self).__init__(offset=offset)

        if surface:
            self.append(SurfaceTissueLayer())
        if nLayers is None:
            nLayers = np.random.randint(1, 10)
        for layer in range(nLayers):
            self.append(RandomTissueLayer(maxBirefringence=maxBirefringence, heightRange=layerHeightRange))

    def reset(self):
        self.__init__(**self.params)
