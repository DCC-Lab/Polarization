from polarization import TissueRetarder, JonesMatrix
from typing import List
import numpy as np


class TissueLayer:
    def __init__(self, birefringence, opticAxis, scattDensity=0, height=200, label=""):
        """

        :param birefringence: Scalar local birefringence dn
        :param opticAxis: Unitary 3D vector of the optic axis in QU plane (r_V = 0).
        :param scattDensity: Density of scatterers.
        :param height: Thickness in microns
        """
        self.birefringence = birefringence
        self.opticAxis = opticAxis
        self.apparentOpticAxis = None

        self.scattDensity = scattDensity
        self.height = height
        self.nScatterers = self.scattDensity * self.height  # amount of scatterers
        self.zScatterers = np.random.rand(self.nScatterers) * self.height  # list of scatterers position dz
        self.amplitude = np.random.randn(self.nScatterers)
        self.label = label

    def resetScatterers(self):
        h = self.height if self.height > 0 else 0
        self.nScatterers = self.scattDensity * h
        self.zScatterers = np.random.rand(self.nScatterers) * h
        self.amplitude = np.random.randn(self.nScatterers)

    def localRetardanceAt(self, k):
        """ Local phase retardance in rad/um. """
        return 2 * k * self.birefringence

    def retardanceAt(self, k):
        """ Total phase retardance of the layer for each scatterer dz. Shape of (nScat, 3)
        Here dZ is a vector of all scatterers dz. A factor of 2 is added to account for the round trip.
        phi(dZ) = 2 * k * dn * OA * dZ
        where OA is the optic axis orientation.
        """
        return 2 * k * self.birefringence * self.opticAxis.reshape((1, 3)) * self.zScatterers[:, None]

    def onePassJonesAt(self, k) -> JonesMatrix:
        return TissueRetarder(self.localRetardanceAt(k) * self.opticAxis.reshape((3, 1)) * self.height)

    def jonesAt(self, k) -> List[JonesMatrix]:
        """ JonesMatrix of all this layer scatterers for a given wavenumber. Shape of (2, 2, nScat). """
        jones = []
        for scatRetardance in self.retardanceAt(k):
            jones.append(TissueRetarder(scatRetardance))
        return jones

    def setApparentOpticAxis(self, aboveRetarder):
        """ From Normalized retardance vector in SO(3),
        Solve Jones expression (z1 and z2) to retrieve the 2 possible jones vector solution e and f.
        Discarding f as they both point to the same stokes.
        Apply above retarder to it.
        Transform back to Stokes vector.
        """
        if np.array_equal(self.opticAxis, np.zeros(3, )):
            self.apparentOpticAxis = self.opticAxis
        else:
            Q, U, V = self.opticAxis
            z1 = np.sqrt((1+Q)/2)
            if z1 == 0:
                z2 = 1
            else:
                z2 = U/(2*z1) - 1j*V/(2*z1)

            e = np.array([z1, z2])
            e_apparent = np.einsum('ij,j', aboveRetarder.T, e)
            z1, z2 = e_apparent[0], e_apparent[1]
            self.apparentOpticAxis = np.asarray([np.abs(z1) ** 2 - np.abs(z2) ** 2,
                                                 2 * np.real(z1 * np.conj(z2)),
                                                 -2 * np.imag(z1 * np.conj(z2))])


class SurfaceTissueLayer(TissueLayer):
    def __init__(self, scattDensity=1000, height=1):
        super(SurfaceTissueLayer, self).__init__(0, np.zeros(3), scattDensity, height)


class RandomTissueLayer(TissueLayer):
    def __init__(self, max_dn=0.0042):
        layerHeight = np.random.randint(60, 400)  # um
        scattDensity = np.random.randint(1, 20)

        birefringence = np.random.uniform(0, max_dn)

        opticAxis = np.random.normal(size=(3,))
        opticAxis[2] = 0  # only birefringent in Q/U planes
        opticAxis /= np.sqrt(np.sum(opticAxis**2))  # now np.sum(opticAxis**2) == 1

        super(RandomTissueLayer, self).__init__(birefringence, opticAxis, height=layerHeight, scattDensity=scattDensity)


class EmptyTissueLayer(TissueLayer):
    def __init__(self, height=2000):
        super(EmptyTissueLayer, self).__init__(birefringence=0, opticAxis=np.zeros(3), height=height, scattDensity=0)


if __name__ == '__main__':
    np.random.seed(528)
    layer = RandomTissueLayer()
    layer.birefringence = 0.0042
    k = 2 * np.pi / 1.3

    print(layer.zScatterers.shape)
    print(layer.opticAxis)
    print(layer.retardanceAt(k).shape)
    print(layer.localRetardanceAt(k))
    print(layer.retardanceAt(k)[:5])
