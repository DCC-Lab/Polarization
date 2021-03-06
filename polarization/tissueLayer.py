from .jonesmatrix import *
from .jonesvector import *
from .pulse import Pulse
from typing import List
import numpy as np

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
        self.scatterers = ScattererGroup(self.thickness, self.scattDensity)

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
            return np.pi / 2 / 2
        else:
            return np.arctan(self.opticAxis[1] / self.opticAxis[0]) / 2

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

        for scat in self.scatterers:
            phi = 2 * scat.dz * K
            dX += scat.strength * exp(1j * phi)
            dY += scat.strength * exp(1j * phi * (1 + self.birefringence))
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


class Scatterer:
    def __init__(self, max_dz):
        self.dz = np.random.rand() * max_dz
        self.strength = np.random.rand()
        self.transferMatrix = None


class ScattererGroup:
    def __init__(self, length, density):
        self.length = length
        self.N = int(density * length)

        self.scatterers = None
        self.reset()

    def reset(self):
        self.scatterers = []
        for _ in range(self.N):
            self.scatterers.append(Scatterer(self.length))

    @property
    def dz(self) -> list:
        return [scatterer.dz for scatterer in self.scatterers]

    @property
    def strength(self) -> list:
        return [scatterer.strength for scatterer in self.scatterers]

    def __iter__(self):
        return iter(self.scatterers)

    def __len__(self):
        return len(self.scatterers)


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
