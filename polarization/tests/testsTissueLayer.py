import envtest
from polarization.tissueLayer import *


class TestTissueLayer(envtest.MyTestCase):
    def testLayerProperties(self):
        pass

    def testTransferMatrix(self):
        pass


class TissueLayerReference:
    """ Old code reference from Martin to validate our new approach. """
    def __init__(self, birefringence, opticAxis, height):
        self.birefringence = birefringence
        self.opticAxis = opticAxis
        self.height = height

    def transferMatrix(self, k, dz=None):
        if dz is None:
            dz = self.height
        ret = k * self.birefringence * self.opticAxis.reshape((3, 1)) * dz
        return self.retarderMatrix(ret)

    def retarderMatrix(self, retardance, diattenuation=None):
        if diattenuation is None:
            diattenuation = np.zeros(retardance.shape)
        dim = retardance.shape
        f = (diattenuation - 1j * retardance) / 2
        c = np.sqrt(np.sum(f ** 2, axis=0)).reshape(1, -1)
        sinch = np.sinh(c) / c
        sinch[c == 0] = 1
        jonesMat = array([[1], [0], [0], [1]]) * (np.cosh(c)) + sinch * (
                array([[1], [0], [0], [-1]]) * f[0, :].reshape(1, -1) +
                array([[0], [1], [1], [0]]) * f[1, :].reshape(1, -1) +
                array([[0], [1j], [-1j], [0]]) * f[2, :].reshape(1, -1))
        if np.size(retardance) == 3:
            M = jonesMat.reshape((2, 2))
        else:
            M = np.squeeze(jonesMat.reshape(2, 2, dim[1], -1))
        return M


if __name__ == '__main__':
    envtest.main()
