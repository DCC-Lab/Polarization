import envtest
from polarization.tissue import *
from polarization.tissueStack import *
from polarization.tissueLayer import *
from polarization.pulse import *

np.random.seed(521)


def extractSV(S1, S2, S3):
    SV1 = np.concatenate(
        (S1[:, 0::2, np.newaxis], S2[:, 0::2, np.newaxis],
         S3[:, 0::2, np.newaxis]), axis=2)
    SV2 = np.concatenate(
        (S1[:, 1::2, np.newaxis], S2[:, 1::2, np.newaxis],
         S3[:, 1::2, np.newaxis]), axis=2)

    return SV1, SV2


def interweave(A, B, axis=0):
    if axis != 0:
        A = np.moveaxis(A, axis, 0)
        B = np.moveaxis(B, axis, 0)

    outShape = list(A.shape)
    outShape[0] *= 2

    C = np.empty(outShape, dtype=A.dtype)
    C[0::2] = A
    C[1::2] = B

    if axis != 0:
        C = np.moveaxis(C, 0, axis)
    return C


def euclideanNorm(arr, axis=0):
    return np.expand_dims(np.linalg.norm(arr, axis=axis), axis=axis)


def computeStokesFromFringes(fringes):
    """ From fringes of shape (4, N*w, z) in k-space to Stokes vectors in z-space """
    tomo = np.fft.fft(fringes, axis=-1)
    return computeStokesFromTomo(tomo)


def computeStokesFromTomo(tomo, weaveAxis=1):
    """ From tomogram of shape (4, N*w, z) in z-space to Stokes vectors in z-space """
    tomv1, tomh1, tomv2, tomh2 = tomo

    tomoX = interweave(tomv1, tomv2, weaveAxis-1)  # (N*w*2, z)
    tomoY = interweave(tomh1, tomh2, weaveAxis-1)

    return computeStokesFromTomoXTomoY(tomoX, tomoY)


def computeStokesFromTomoXTomoY(tomoX, tomoY):
    S0 = np.absolute(tomoX) ** 2 + np.absolute(tomoY) ** 2
    S1 = np.absolute(tomoX) ** 2 - np.absolute(tomoY) ** 2
    S2 = 2 * np.real(tomoX * np.conj(tomoY))
    S3 = -2 * np.imag(tomoX * np.conj(tomoY))

    return np.asarray([S0, S1, S2, S3])


class TestPSSignal(envtest.MyTestCase):
    @staticmethod
    def stokesFromFringes(fringes):
        stokes = computeStokesFromFringes(fringes)
        S0, S1, S2, S3 = np.moveaxis(stokes, 1, 2)

        SV1, SV2 = extractSV(S1, S2, S3)
        SV1 = SV1 / euclideanNorm(SV1, axis=2)
        SV2 = SV2 / euclideanNorm(SV2, axis=2)

        x = np.concatenate([SV1, SV2], axis=2)
        return np.moveaxis(x, 0, 2)

    @staticmethod
    def createTestScan(layers):
        resolution = 150
        centerWavelength = 1.3
        bandwidth = 0.13
        Dk = 2 * np.pi / centerWavelength ** 2 * bandwidth
        height = np.pi / Dk * resolution

        tissue = TissueTestFromLayers(layers, height=int(height))
        tissue.display()

        pIn = PulseCollection.dualInputStates(centerWavelength, bandwidth, resolution=resolution)

        return tissue.scan(pIn, verbose=True)

    def runTestScan(self, layers):
        pOut = self.createTestScan(layers)
        stokes = self.stokesFromFringes(pOut.fringes)

        fig, axes = plt.subplots(1, 6)
        for i, ax in enumerate(axes):
            ax.imshow(stokes[:, i].T, aspect='auto', cmap='gray')
        plt.show()

    def testTissueUnit(self):
        layers = [TissueLayer(0.0005, (1, -1, 0), 12, 200),
                  TissueLayer(0.001, (0.2, 1, 0), 8, 200),
                  TissueLayer(0.002, (-1, 0.3, 0), 3, 200),  # problematic layer
                  TissueLayer(0.0005, (0.8, -1, 0), 20, 200)]

        # self.runTestScan(layers)
        self.fail()

    def testTissueSameLayers(self):
        layers = [TissueLayer(0.001, (0.2, 1, 0), 8, 200),
                  TissueLayer(0.001, (0.2, 1, 0), 8, 200),
                  TissueLayer(0.001, (0.2, 1, 0), 8, 200),
                  TissueLayer(0.001, (0.2, 1, 0), 8, 200)]

        # self.runTestScan(layers)

    def testTissueScattering(self):
        layers = [TissueLayer(0.001, (0.2, 1, 0), 12, 200),
                  TissueLayer(0.001, (0.2, 1, 0), 8, 200),
                  TissueLayer(0.001, (0.2, 1, 0), 3, 200),
                  TissueLayer(0.001, (0.2, 1, 0), 20, 200)]

        # self.runTestScan(layers)

    def testTissueBirefringence(self):
        layers = [TissueLayer(0.0005, (0.2, 1, 0), 8, 200),
                  TissueLayer(0.001, (0.2, 1, 0), 8, 200),
                  TissueLayer(0.002, (0.2, 1, 0), 8, 200),
                  TissueLayer(0.0005, (0.2, 1, 0), 8, 200)]

        # self.runTestScan(layers)

    def testTissueOA(self):
        layers = [TissueLayer(0.001, (1, -1, 0), 8, 200),
                  TissueLayer(0.001, (0.2, 1, 0), 8, 200),
                  TissueLayer(0.001, (-1, 0.3, 0), 8, 200),
                  TissueLayer(0.001, (0.8, -1, 0), 8, 200)]

        # self.runTestScan(layers)
        self.fail()

    def testTissueOASimple(self):
        layers = [TissueLayer(0.001, (1, -1, 0), 8, 200),
                  TissueLayer(0.001, (0, 1, 0), 8, 200),
                  TissueLayer(0.001, (-1, 0, 0), 8, 200),
                  TissueLayer(0.001, (0, -1, 0), 8, 200)]

        # self.runTestScan(layers)
        self.fail()

    def testTissueOAx(self):
        layers = [TissueLayer(0.001, (1, 0, 0), 8, 200),
                  TissueLayer(0.001, (-1, 0, 0), 8, 200),
                  TissueLayer(0.001, (1, 0, 0), 8, 200),
                  TissueLayer(0.001, (-1, 0, 0), 8, 200)]

        # self.runTestScan(layers)

    def testTissueOAy(self):
        layers = [TissueLayer(0.001, (0, 1, 0), 8, 200),
                  TissueLayer(0.001, (0, -1, 0), 8, 200),
                  TissueLayer(0.001, (0, 1, 0), 8, 200),
                  TissueLayer(0.001, (0, -1, 0), 8, 200)]

        # self.runTestScan(layers)

    def testTissueOASpin(self):
        """ Classic discontinuity around 3rd layer. """
        layers = [TissueLayer(0.001, (1, 0, 0), 8, 200),
                  TissueLayer(0.001, (0, 1, 0), 8, 200),
                  TissueLayer(0.001, (-1, 0, 0), 8, 200),
                  TissueLayer(0.001, (0, -1, 0), 8, 200)]

        # self.runTestScan(layers)
        self.fail()

    def testTissueOASpin3Layers(self):
        """ Discontinuity only before last layer. """
        layers = [TissueLayer(0.001, (0, 1, 0), 8, 200),
                  TissueLayer(0.001, (-1, 0, 0), 8, 200),
                  TissueLayer(0.001, (0, -1, 0), 8, 200)]

        # self.runTestScan(layers)
        self.fail()

    def testTissueOASpin2Layers(self):
        """ No Discontinuity. """
        layers = [TissueLayer(0.001, (-1, 0, 0), 8, 200),
                  TissueLayer(0.001, (0, -1, 0), 8, 200)]

        # self.runTestScan(layers)


class TissueTest(RandomTissue2D):
    def __init__(self, height):
        layers = [TissueLayer(0.002, (0.1, 1, 0), 1000, 10), TissueLayer(0.0005, (1, -1, 0), 12, 600),
                  TissueLayer(0.001, (0.2, 1, 0), 8, 500), TissueLayer(0.001, (-1, 0.3, 0), 3, 400),
                  TissueLayer(0.0005, (0.8, -1, 0), 20, 500), TissueLayer(0.002, (1, 0.8, 0), 15, 300)]
        testStack = TissueStack(offset=400, layers=layers)
        super(TissueTest, self).__init__(referenceStack=testStack, width=40, flat=True, height=height)


class TissueTestUnit(RandomTissue2D):
    def __init__(self, height):
        layers = [TissueLayer(0.0005, (1, -1, 0), 12, 200),
                  TissueLayer(0.001, (0.2, 1, 0), 8, 200),
                  TissueLayer(0.002, (-1, 0.3, 0), 3, 200),  # problematic layer
                  TissueLayer(0.0005, (0.8, -1, 0), 20, 200)]
        testStack = TissueStack(offset=100, layers=layers)
        super(TissueTestUnit, self).__init__(referenceStack=testStack, width=20, flat=True, height=height)


class TissueTestFromLayers(RandomTissue2D):
    def __init__(self, layers, height):
        testStack = TissueStack(offset=100, layers=layers)
        super(TissueTestFromLayers, self).__init__(referenceStack=testStack, width=20, flat=True, height=height)


if __name__ == '__main__':
    envtest.main()
