import envtest
from polarization.tissue import *
from polarization.tissueStack import *
from polarization.tissueLayer import *
from polarization.pulse import *

np.random.seed(521)


class TestTissue(envtest.MyTestCase):
    def setUp(self) -> None:
        pass

    def testRandomTissue2DMap(self):
        tissue = RandomTissue2D(nLayers=6)
        self.assertEqual(tissue._layerSizeMap.shape, (7, 200))

    def testPSOCTFringes(self):
        resolution = 5
        centerWavelength = 1.3
        bandwidth = 0.13
        tissue = RandomTissue2D(width=2, surface=False, nLayers=1, layerHeightRange=(100, 110), offset=200)
        pIn = PulseCollection.dualInputStates(centerWavelength, bandwidth, resolution=resolution)

        pOut = tissue.scan(pIn)

        for pulseState in pOut:
            ExIsAllZeros = not np.any(pulseState.Ex)
            EyIsAllZeros = not np.any(pulseState.Ey)
            self.assertFalse(ExIsAllZeros)
            self.assertFalse(EyIsAllZeros)

    def testPSOCT(self):
        resolution = 256
        centerWavelength = 1.3
        bandwidth = 0.13

        tissue = TissueTestUnit()
        pIn = PulseCollection.dualInputStates(centerWavelength, bandwidth, resolution=resolution)

        pOut = tissue.scan(pIn, verbose=True)
        pOut.display()


class TissueTestUnit(RandomTissue2D):
    def __init__(self):
        layers = [TissueLayer(0.004, (0, 1, 0), 20, 300), TissueLayer(0.004, (1, 0, 0), 1, 200)]
        testStack = TissueStack(offset=300, layers=layers)
        super(TissueTestUnit, self).__init__(referenceStack=testStack, width=4, flat=True)


if __name__ == '__main__':
    envtest.main()
