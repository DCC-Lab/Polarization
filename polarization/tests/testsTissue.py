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
        self.assertEqual(tissue.map.shape, (7, 200))

    def testPSOCT(self):
        resolution = 50
        centerWavelength = 1.3
        bandwidth = 0.13

        tissue = TissueTestUnit()
        pIn = PulseCollection.dualInputStates(centerWavelength, bandwidth, resolution=resolution)

        # fixme: The computations are way too slow (x400 ish). Parallelization will be required.
        #  All objects are currently immutable.
        pOut = tissue.scan(pIn, verbose=True)
        pOut.display()


class TissueTestUnit(RandomTissue2D):
    def __init__(self):
        layers = [TissueLayer(0.004, (0, 1, 0), 20, 80), TissueLayer(0.004, (1, 0, 0), 1, 80)]
        testStack = TissueStack(offset=80, layers=layers)
        super(TissueTestUnit, self).__init__(referenceStack=testStack, width=2, flat=True)


if __name__ == '__main__':
    envtest.main()
