import envtest
import numpy as np
from polarization.tissue import *
from polarization.pulse import *
import matplotlib.pyplot as plt

np.random.seed(521)


class TestTissue(envtest.MyTestCase):
    def setUp(self) -> None:
        pass

    def testRandomTissue2DMap(self):
        tissue = RandomTissue2D(nLayers=6)
        self.assertEqual(tissue.map.shape, (7, 200))

    def testTissueStackAt(self):
        tissue = RandomTissue2D(nLayers=6)

        stack = tissue.stackAt(5)
        self.assertAlmostEqual(len(stack), 6)

    def testPSOCT(self):
        resolution = 20
        centerWavelength = 1.3
        bandwidth = 0.13

        tissue = RandomTissue2D(width=2, surface=False, nLayers=1, layerHeightRange=(100, 110), offset=200)
        pIn = PulseCollection.dualInputStates(centerWavelength, bandwidth, resolution=resolution)

        # fixme: The computations are way too slow (x400 ish). Parallelization will be required.
        #  All objects are currently immutable.
        pOut = tissue.scan(pIn, verbose=True)

        pOut.display()
