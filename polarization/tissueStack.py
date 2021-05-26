from .tissueLayer import *
from .jonesvector import JonesVector
from .jonesmatrix import JonesMatrix, Vacuum
from .pulse import *
from typing import List
import numpy as np

__all__ = ['TissueStack', 'RandomTissueStack']


class TissueStack:
    def __init__(self, offset=0, layers=None):
        """
        A stack of multiple tissue layers along depth axis.

        :param offset: Surface offset in microns before the first TissueLayer
        :param layers: Layers to stack.
        """
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

    def initTransferMatrices(self):
        self.forwardMatrices = [Vacuum(physicalLength=self.offset)]
        for i, layer in enumerate(self.layers[:-1]):
            self.forwardMatrices.append(self.forwardMatrices[i] * layer.transferMatrix())

        self.backwardMatrices = [Vacuum(physicalLength=self.offset)]
        for i, layer in enumerate(self.layers[:-1]):
            self.backwardMatrices.append(layer.transferMatrix().backward() * self.backwardMatrices[i])

    def transferMatrix(self, layerIndex=None, backward=False):
        if backward:
            return self.backwardMatrices[layerIndex]
        else:
            return self.forwardMatrices[layerIndex]

        # M = JonesMatrix(1, 0, 0, 1)
        # if backward:
        #     backwardLayers = self.layers[:layerIndex]
        #     backwardLayers.reverse()
        #     for layer in backwardLayers:
        #         M *= layer.transferMatrix().backward()
        #     M *= Vacuum(physicalLength=self.offset)
        # else:
        #     M *= Vacuum(physicalLength=self.offset)
        #     for layer in self.layers[: layerIndex]:
        #         M *= layer.transferMatrix()

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

    def backscatterMany(self, vectors):
        if type(vectors) is PulseCollection:
            return self._backscatterPulseCollection(vectors)
        elif type(vectors) is Pulse:
            return self._backscatterPulse(vectors)
        else:
            return self._backscatterVectors(vectors)

    def _backscatterVectors(self, vectors: List[JonesVector]):
        K = [v.k for v in vectors]
        self.initBackscatteringAt(K)

        vectorsOut = []
        for v in vectors:
            vectorsOut.append(self.backscatter(v))
        
        if type(vectors) is Pulse:
            return Pulse(vectors=vectorsOut)
        return vectorsOut

    def _backscatterPulse(self, pulse: Pulse):
        return self._backscatterPulseCollection(PulseCollection(pulses=[pulse]))[0]

    def _backscatterPulseCollection(self, pulses: PulseCollection):
        self.initBackscatteringAt(pulses.k)

        self.initTransferMatrices()

        vectorsOut = [[] for _ in pulses]
        for i, k in enumerate(pulses.k):
            vectorOut = [JonesVector(0, 0, k=k) for _ in pulses]
            for j, layer in enumerate(self.layers):
                M = self.transferMatrix(j, backward=True) * self.transferMatrix(j) * layer.backscatteringMatrixAt(k)
                for p, pulse in enumerate(pulses):
                    vectorOut[p] += M * pulse.vectors[i]
            for p in range(len(pulses)):
                vectorsOut[p].append(vectorOut[p])

        pulsesOut = [Pulse(vectors) for vectors in vectorsOut]
        return pulsesOut

    def initBackscatteringAt(self, K):
        for layer in self.layers:
            layer.initBackscatteringMatrixAt(K)


class RandomTissueStack(TissueStack):
    def __init__(self, surface=True, maxBirefringence=0.0042, nLayers=None, offset=None, layerHeightRange=(60, 400)):
        """
        Generate a random TissueStack.

        Optional arguments:
            surface: Add a thin and highly scattering surface layer to mimic Fresnel reflection.
            maxBirefringence: Maximum birefringence (delta n) of any generated layers.
            nLayers: Amount of layers to generate. Default is random(1, 10).
            offset: Surface offset in microns before the first layer. Default is random(200, 600).
            layerHeighRange: Range in microns from which to pick layer thicknesses. Default is (60, 400).
        """
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
