from .jonesmatrix import *
from .jonesvector import *
from .pulse import Pulse
from typing import List
import numpy as np


class TissueLayer:
    def __init__(self, birefringence, opticAxis, scattDensity=0, thickness=200):
        """

        :param birefringence: Scalar local birefringence dn
        :param opticAxis: Unitary 3D vector of the optic axis in QUV plane (r_V = 0).
        :param scattDensity: Density of scatterers.
        :param thickness: Thickness in microns
        """

        self.birefringence = birefringence
        self.opticAxis = opticAxis
        self.scattDensity = scattDensity
        self.thickness = thickness

        self.apparentOpticAxis = None
        self.scatterers = ScattererGroup(self.thickness, self.scattDensity)

    @property
    def opticAxis(self):
        return self._opticAxis

    @opticAxis.setter
    def opticAxis(self, vector):
        """ Assert unitary vector """
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
            return BirefringentMaterial(deltaIndex=self.birefringence, fastAxisOrientation=self.orientation,
                                        physicalLength=self.thickness)
        else:
            return BirefringentMaterial(deltaIndex=self.birefringence, fastAxisOrientation=self.orientation,
                                        physicalLength=dz)

    def propagateThrough(self, vector: JonesVector) -> JonesVector:
        return self.transferMatrix() * vector

    def propagateManyThrough(self, vectors: List[JonesVector]) -> List[JonesVector]:
        J = []
        for v in vectors:
            J.append(self.propagateThrough(v))
        return J

    def backscatter(self, vector: JonesVector) -> JonesVector:
        signal = JonesVector(0, 0, k=vector.k)
        for scat in self.scatterers:
            scatSignal = self.transferMatrix(dz=2*scat.dz) * vector * scat.strength
            signal += scatSignal
        return signal

    def backscatterMany(self, vectors: List[JonesVector]) -> List[JonesVector]:
        vectorsOut = []
        for v in vectors:
            vectorsOut.append(self.backscatter(v))
        return vectorsOut


class Scatterer:
    def __init__(self, max_dz):
        self.dz = np.random.rand() * max_dz
        self.strength = np.random.rand()


class ScattererGroup:
    def __init__(self, length, density):
        self.length = length
        self.N = int(density * length)

        self.scatterers = None
        self.resetScatterers()

    def resetScatterers(self):
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
    def __init__(self, maxBirefringence=0.0042):
        layerHeight = np.random.randint(60, 400)  # um
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
