from .jonesmatrix import *
from .jonesvector import *
from .pulse import Pulse
from typing import List
import numpy as np
from collections import namedtuple

__all__ = ['TissueLayer', 'SurfaceTissueLayer', 'RandomTissueLayer', 'EmptyTissueLayer']


class TissueLayer:
    def __init__(self, birefringence, opticAxis, scattDensity=0, thickness=200):
        """
        A single layer of birefringent material.

        :param birefringence: Scalar local birefringence dn
        :param opticAxis: Unitary 3D vector of the optic axis in QUV plane (r_V = 0).
        :param scattDensity: Density of scatterers.
        :param thickness: Thickness in microns
        """

        self.birefringence = birefringence
        self.opticAxis = opticAxis
        self.scattDensity = scattDensity
        self.thickness = thickness

        self.cachedMatrices = {}
        self.apparentOpticAxis = None
        self.scatterers = Scatterers(self.thickness, self.scattDensity)

    def copy(self, thickness=None):
        if thickness is None:
            thickness = self.thickness
        return TissueLayer(birefringence=self.birefringence, opticAxis=self.opticAxis,
                           scattDensity=self.scattDensity, thickness=thickness)

    @property
    def opticAxis(self):
        return self._opticAxis

    @opticAxis.setter
    def opticAxis(self, vector):
        """ Assert unitary vector """
        vector = np.asarray(vector, dtype=float)
        if np.sum(vector**2) != 0:
            vector /= np.sqrt(np.sum(vector ** 2))
        self._opticAxis = vector

    @property
    def orientation(self):
        """ Orientation in radians in X/Y coordinates from the Q/U plane of linear polarization. """
        if self.opticAxis[0] == 0:
            return np.pi / 4
        else:
            return np.arctan2(self.opticAxis[1], self.opticAxis[0]) / 2

    def transferMatrix(self, dz=None) -> BirefringentMaterial:
        if dz is None:
            dz = self.thickness
        return BirefringentMaterial(deltaIndex=self.birefringence, fastAxisOrientation=self.orientation, physicalLength=dz)

    def propagateThrough(self, vector: JonesVector) -> JonesVector:
        return self.transferMatrix() * vector

    def propagateManyThrough(self, vectors: List[JonesVector]) -> List[JonesVector]:
        J = []
        for v in vectors:
            J.append(self.propagateThrough(v))
        return J

    def backscatter(self, vector: JonesVector) -> JonesVector:
        return self.backscatteringMatrixAt(vector.k) * vector

    def backscatterMany(self, vectors):
        vectorsOut = []
        if type(vectors) is Pulse:
            K = vectors.k
        else:
            K = [v.k for v in vectors]
        self.initBackscatteringMatrixAt(K)

        for v in vectors:
            vectorsOut.append(self.backscatter(v))

        if type(vectors) is Pulse:
            return Pulse(vectors=vectorsOut)
        return vectorsOut

    def initBackscatteringMatrixAt(self, K):
        dX, dY = self.scatteringDeltaAt(K)
        for i, k in enumerate(K):
            self.cachedMatrices[k] = JonesMatrix(A=dX[i], B=0, C=0, D=dY[i], orientation=self.orientation)

    def scatteringDeltaAt(self, K):
        dX, dY = 0, 0
        if type(K) is list:
            dX, dY = np.zeros(len(K), dtype=complex), np.zeros(len(K), dtype=complex)
            K = np.asarray(K)

        for dz, strength in zip(self.scatterers.dz, self.scatterers.strength):
            phi = 2 * dz * K
            dX += strength * exp(1j * phi)
            dY += strength * exp(1j * phi * (1 + self.birefringence))
        return dX, dY

    def backscatteringMatrixAt(self, k):
        matrix = self.cachedMatrices.get(k)
        if matrix is None:
            dX, dY = self.scatteringDeltaAt(k)
            return JonesMatrix(A=dX, B=0, C=0, D=dY, orientation=self.orientation)
        else:
            return matrix

    def resetScatterers(self):
        self.cachedMatrices = {}
        return self.scatterers.reset()


Scatterer = namedtuple('Scatterer', ['dz', 'strength'])


class RandomScatterer:
    def __new__(cls, max_dz):
        return Scatterer(dz=np.random.rand() * max_dz, strength=np.random.rand())


class Scatterers:
    def __init__(self, length, density=None, N=None):
        self.length = length
        if density is not None:
            self.N = int(density * length)
        else:
            self.N = N

        self.dz = []
        self.strength = []
        self.reset()

    def reset(self):
        if self.N > 0:
            self.dz = np.random.rand(self.N) * self.length
            self.strength = np.random.rand(self.N)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> Scatterer:
        if self.n < self.N:
            scatterer = Scatterer(self.dz[self.n], self.strength[self.n])
            self.n += 1
            return scatterer
        else:
            raise StopIteration

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        return Scatterer(self.dz[item], self.strength[item])


class SurfaceTissueLayer(TissueLayer):
    def __init__(self, scattDensity=1000, thickness=1):
        super(SurfaceTissueLayer, self).__init__(birefringence=0, opticAxis=np.zeros(3),
                                                 scattDensity=scattDensity, thickness=thickness)


class RandomTissueLayer(TissueLayer):
    def __init__(self, maxBirefringence=0.0042, heightRange=(60, 400)):
        layerHeight = np.random.randint(*heightRange)  # um
        scattDensity = np.random.randint(1, 20)

        birefringence = np.random.uniform(0, maxBirefringence)

        opticAxis = np.random.normal(size=(3,))
        opticAxis[2] = 0  # only birefringent in Q/U planes

        super(RandomTissueLayer, self).__init__(birefringence=birefringence, opticAxis=opticAxis,
                                                thickness=layerHeight, scattDensity=scattDensity)


class EmptyTissueLayer(TissueLayer):
    def __init__(self, thickness=2000):
        super(EmptyTissueLayer, self).__init__(birefringence=0, opticAxis=np.zeros(3),
                                               thickness=thickness, scattDensity=0)
