import envtest
from polarization.tissueLayer import *


class TestTissueLayer(envtest.MyTestCase):
    def setUp(self) -> None:
        self.birefringence = 0.004
        self.opticAxis = np.asarray((2, 1, 0), dtype=np.float)
        self.opticAxis /= np.sqrt(np.sum(self.opticAxis**2))
        self.scattDensity = 10
        self.thickness = 200

        self.layer = TissueLayer(self.birefringence, self.opticAxis, self.scattDensity, self.thickness)
        self.layerRef = TissueLayerReference(self.birefringence, self.opticAxis, self.scattDensity, self.thickness)

    def testLayerProperties(self):
        pass

    def testSymetricTransferMatrix(self):
        """ Shows how to construct a symmetric retarder that exactly fits the old code. """
        k = 1.3
        delta = k * self.birefringence * self.thickness
        J = JonesMatrix(A=exp(-1j * delta/2), B=0, C=0, D=exp(1j * delta/2))
        J.orientation = self.layer.orientation

        MRef = self.layerRef.transferMatrix(k=k)

        for a, b in zip([J.A, J.B, J.C, J.D], np.nditer(MRef)):
            self.assertAlmostEqual(a, b, 10)

    def testPropagationTransferMatrixNotEqual(self):
        """ Not equal because we are not defining our retarder matrix as symmetric. """
        k = 1.3
        M = self.layer.transferMatrix().computeMatrix(k=k)

        MRef = self.layerRef.transferMatrix(k=k)

        for a, b in zip(np.nditer(M), np.nditer(MRef)):
            self.assertNotAlmostEqual(a, b, 10)

    def testPropagationWithPhaseSymetricMatrix(self):
        k = 1.3
        pIn = JonesVector.leftCircular()
        pIn.k = k
        delta = k * self.birefringence * self.thickness
        J = JonesMatrix(A=exp(-1j * delta / 2), B=0, C=0, D=exp(1j * delta / 2))
        J.orientation = self.layer.orientation

        pOut = J * pIn

        MRef = self.layerRef.transferMatrix(k=k)
        pOutRef = np.reshape(np.einsum('ij, j', MRef, np.asarray([pIn.Ex, pIn.Ey])), (2,))

        self.assertAlmostEqual(pOut.Ex, pOutRef[0])
        self.assertAlmostEqual(pOut.Ey, pOutRef[1])

    def testPropagation(self):
        """ When using our not phase-symmetric matrix, we expect same output orientation, but different phase."""
        k = 1.3
        pIn = JonesVector.leftCircular()
        pIn.k = k

        pOut = self.layer.transferMatrix() * pIn

        MRef = self.layerRef.transferMatrix(k=k)
        pOutRef = np.reshape(np.einsum('ij, j', MRef, np.asarray([pIn.Ex, pIn.Ey])), (2,))
        pOutRef = JonesVector(pOutRef[0], pOutRef[1])

        self.assertNotAlmostEqual(pOut.Ex, pOutRef.Ex)
        self.assertNotAlmostEqual(pOut.Ey, pOutRef.Ey)

        self.assertEqual(pOut.orientation, pOutRef.orientation)


class TissueLayerReference:
    """ Old code reference from Martin to validate our new approach.
    Independent from this API.
    """
    def __init__(self, birefringence, opticAxis, scattDensity, thickness):
        self.birefringence = birefringence
        self.opticAxis = opticAxis
        self.scattDensity = scattDensity
        self.thickness = thickness

    def transferMatrix(self, k, dz=None):
        if dz is None:
            dz = self.thickness
        retardance = k * self.birefringence * self.opticAxis.reshape((3, 1)) * dz
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
