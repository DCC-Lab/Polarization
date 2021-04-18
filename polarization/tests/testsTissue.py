import envtest
import numpy as np
from polarization.tissue import *
from polarization.tissueStack import *
from polarization.tissueLayer import *
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
        # 103 seconds at res = 100, scat = 100*10 ?, 330s at 512 res
        resolution = 50
        centerWavelength = 1.3
        bandwidth = 0.13

        tissue = TissueTestUnit()  # 105s
        pIn = PulseCollection.dualInputStates(centerWavelength, bandwidth, resolution=resolution)

        # fixme: The computations are way too slow (x400 ish). Parallelization will be required.
        #  All objects are currently immutable.
        pOut = tissue.scan(pIn, verbose=True)

        pOut.display()

        # fixme Tissue.stackAt :  use the same scatterer at each position


class TissueTestUnit(RandomTissue2D):
    def __init__(self):
        layers = [TissueLayer(0.004, (0, 1, 0), 20, 80), TissueLayer(0.004, (1, 0, 0), 1, 80)]
        testStack = TissueStack(offset=80, layers=layers)
        super(TissueTestUnit, self).__init__(referenceStack=testStack, width=2, flat=True)
