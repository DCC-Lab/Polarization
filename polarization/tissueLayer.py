from polarization import JonesMatrix, PhaseRetarder, JonesVector, Pulse
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

    def transferMatrixAt(self, k: float, dz=None) -> PhaseRetarder:
        if dz is None:
            M = PhaseRetarder(delta=k * self.birefringence * self.thickness)
        else:
            M = PhaseRetarder(delta=k * self.birefringence * dz * 2)
        M.orientation = self.orientation
        return M

    def propagateThrough(self, pulse: Pulse) -> Pulse:
        # also allow a list of JonesVector at input
        return self.transferMatrixAt(pulse.kc) * pulse

    def scatterRoundTrip(self, pulse: Pulse) -> Pulse:
        # also allow a list of JonesVector at input
        J_L = []
        for scat in self.scatterers:
            J_s = self.transferMatrixAt(pulse.kc, dz=scat.dz) * scat.strength
            J_K = []
            for k in pulse.kSpectrum:
                J_K.append(J_s * np.exp(1j * k * 2 * (scat.dz + self.position)))
            J_L.append(J_K)

        for i in range(len(pulse.kStates)):
            J_sum = JonesMatrix(0, 0, 0, 0)
            for j in range(len(self.scatterers)):
                J_sum += J_L[j][i]
            pulse.kStates[i] = J_sum * pulse.kStates[i]

        return pulse

    def setApparentOpticAxis(self):
        # move to stack ? makes no sense without a stack
        pass


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

    print(layer.zScatterers.shape)
    print(layer.opticAxis)

    # def setApparentOpticAxis(self, aboveRetarder):
    #     """ From Normalized retardance vector in SO(3),
    #     Solve Jones expression (z1 and z2) to retrieve the 2 possible jones vector solution e and f.
    #     Discarding f as they both point to the same stokes.
    #     Apply above retarder to it.
    #     Transform back to Stokes vector.
    #     """
    #     if np.array_equal(self.opticAxis, np.zeros(3, )):
    #         self.apparentOpticAxis = self.opticAxis
    #     else:
    #         z1, z2 = SO3_to_SU2(self.opticAxis)
    #         e = np.array([z1, z2])
    #         e_apparent = np.einsum('ij,j', aboveRetarder.T, e)
    #         z1, z2 = e_apparent[0], e_apparent[1]
    #         self.apparentOpticAxis = np.asarray([np.abs(z1) ** 2 - np.abs(z2) ** 2,
    #                                              2 * np.real(z1 * np.conj(z2)),
    #                                              -2 * np.imag(z1 * np.conj(z2))])
    #
    # def resetScatterers(self):
    #     h = self.thickness if self.thickness > 0 else 0
    #     self.nScatterers = self.scattDensity * h
    #     self.zScatterers = np.random.rand(self.nScatterers) * h
    #     self.amplitude = np.random.randn(self.nScatterers)
