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

        self.position = 0
        self.apparentOpticAxis = None
        self.scatterers = [Scatterer(self.thickness)] * int(self.scattDensity * self.thickness)

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
                                        physicalLength=2*dz)

    def propagateThrough(self, vector: JonesVector) -> JonesVector:
        return self.transferMatrix() * vector

    def propagateManyThrough(self, vectors: List[JonesVector]) -> List[JonesVector]:
        J = []
        M = self.transferMatrix()
        for v in vectors:
            J.append(M * v)
        return J

    def backscatter(self, vector: JonesVector) -> JonesVector:
        # todo/test-me : While we use to create J_s from k_c, it will now use the vector's specific k instead.
        J_sum = JonesMatrix(0, 0, 0, 0)
        for scat in self.scatterers:
            J_s = self.transferMatrix(dz=scat.dz) * scat.strength
            J_sum += J_s * np.exp(1j * vector.k * 2 * (scat.dz + self.position))
        return J_sum * vector

    def backscatterMany(self, vectors: List[JonesVector]) -> List[JonesVector]:
        vectorsOut = []
        for i, v in enumerate(vectors):
            vectorsOut.append(self.backscatter(v))
        return vectorsOut


class Scatterer:
    def __init__(self, max_dz):
        self.dz = np.random.rand() * max_dz
        self.strength = np.random.rand()


class SurfaceTissueLayer(TissueLayer):
    def __init__(self, scattDensity=1000, thickness=1):
        super(SurfaceTissueLayer, self).__init__(0, np.zeros(3), scattDensity, thickness)


class RandomTissueLayer(TissueLayer):
    def __init__(self, max_dn=0.0042):
        layerHeight = np.random.randint(60, 400)  # um
        scattDensity = np.random.randint(1, 20)

        birefringence = np.random.uniform(0, max_dn)

        opticAxis = np.random.normal(size=(3,))
        opticAxis[2] = 0  # only birefringent in Q/U planes
        opticAxis /= np.sqrt(np.sum(opticAxis**2))  # now np.sum(opticAxis**2) == 1

        super(RandomTissueLayer, self).__init__(birefringence, opticAxis, thickness=layerHeight, scattDensity=scattDensity)


class EmptyTissueLayer(TissueLayer):
    def __init__(self, thickness=2000):
        super(EmptyTissueLayer, self).__init__(birefringence=0, opticAxis=np.zeros(3), thickness=thickness, scattDensity=0)


if __name__ == '__main__':
    np.random.seed(528)
    layer = RandomTissueLayer()
    layer.birefringence = 0.0042
    k_c = 2 * np.pi / 1.3
