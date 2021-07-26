from .tissueLayer import *
from .jonesvector import JonesVector
from .jonesmatrix import JonesMatrix, Vacuum
from .pulse import *
from typing import List
import numpy as np

__all__ = ['TissueStack', 'RandomTissueStack']


class TissueStack:
    def __init__(self, offset=0, layers=None, height=3000):
        """
        A stack of multiple tissue layers along depth axis.

        :param offset: Surface offset in microns before the first TissueLayer
        :param layers: Layers to stack.
        """
        self.layers: List[TissueLayer] = []
        self.offset = offset
        self.height = height

        self.forwardMatrices = None
        self.backwardMatrices = None

        if layers is not None:
            for layer in layers:
                self.append(layer)

        self._scattDensity = None
        self._opticAxis = None
        self._apparentOpticAxis = None
        self._birefringence = None

    def append(self, layer: TissueLayer):
        self.layers.append(layer)

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def initTransferMatrices(self):
        self.forwardMatrices = [Vacuum(physicalLength=self.offset)]
        self.backwardMatrices = [Vacuum(physicalLength=self.offset)]
        for i, layer in enumerate(self.layers):
            self.forwardMatrices.append(layer.transferMatrix() * self.forwardMatrices[i])
            self.backwardMatrices.append(self.backwardMatrices[i] * layer.transferMatrix().backward())

    def transferMatrix(self, layerIndex=-1, backward=False):
        if self.forwardMatrices is None:
            self.initTransferMatrices()

        if backward:
            return self.backwardMatrices[layerIndex]
        else:
            return self.forwardMatrices[layerIndex]

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
            signal += self.transferMatrix(i, backward=True) * (layer.backscatter(self.transferMatrix(i) * vector))
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
                M = self.transferMatrix(j, backward=True) * layer.backscatteringMatrixAt(k) * self.transferMatrix(j)
                for p, pulse in enumerate(pulses):
                    vectorOut[p] += M * pulse.vectors[i]
            for p in range(len(pulses)):
                vectorsOut[p].append(vectorOut[p])

        pulsesOut = [Pulse(vectors) for vectors in vectorsOut]
        return pulsesOut

    def initBackscatteringAt(self, K):
        for layer in self.layers:
            layer.initBackscatteringMatrixAt(K)

    @property
    def scattDensity(self):
        if self._scattDensity is None:
            self._scattDensity = np.zeros(int(self.height))
            currentPosition = int(self.offset)
            for layer in self.layers:
                self._scattDensity[currentPosition: currentPosition + int(layer.thickness)] = layer.scattDensity
                currentPosition += int(layer.thickness)
        return self._scattDensity

    @property
    def opticAxis(self):
        if self._opticAxis is None:
            self._opticAxis = np.zeros((3, self.height))
            currentPosition = int(self.offset)
            for layer in self.layers:
                self._opticAxis[:, currentPosition: currentPosition + int(layer.thickness)] = layer.opticAxis.reshape((3, 1))
                currentPosition += int(layer.thickness)
        return self._opticAxis

    @property
    def apparentOpticAxis(self):
        if self._apparentOpticAxis is None:
            self._apparentOpticAxis = np.zeros((3, self.height))
            currentPosition = int(self.offset)
            for layer in self.layers:
                self._apparentOpticAxis[:, currentPosition: currentPosition + int(layer.thickness)] = layer.apparentOpticAxis.reshape((3, 1))
                currentPosition += int(layer.thickness)
        return self._apparentOpticAxis

    @property
    def birefringence(self):
        if self._birefringence is None:
            self._birefringence = np.zeros(int(self.height))
            currentPosition = int(self.offset)
            for layer in self.layers:
                self._birefringence[currentPosition: currentPosition + int(layer.thickness)] = layer.birefringence
                currentPosition += int(layer.thickness)
        return self._birefringence


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
