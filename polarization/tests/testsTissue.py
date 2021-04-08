import envtest
import numpy as np
from polarization.tissue import *
from polarization.pulse import Pulse
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

        tissue = RandomTissue2D(width=3, surface=False, nLayers=1, layerHeightRange=(80, 100))
        # todo: implement offset as a empty propagation
        p1 = Pulse.horizontal(centerWavelength, bandwidth, resolution=resolution)
        p2 = Pulse.leftCircular(centerWavelength, bandwidth, resolution=resolution)  # todo: define a pulse collection instead

        # todo: better implement the following in oop
        # this is (too slow)**4
        fringes = np.zeros((tissue.width, 4, resolution), dtype=np.complex)
        for i, stack in enumerate(tissue):
            print(i, 1)
            e1 = stack.backscatterMany(p1)
            print(i, 2)
            e2 = stack.backscatterMany(p2)
            fringes[i] = np.asarray([e1.Ex, e1.Ey, e2.Ex, e2.Ey])

        intensityFringes = 10 * np.log10(np.abs(fringes ** 2))

        plt.imshow(intensityFringes[:, 0])
        plt.show()

        # p1Out = tissue.scan(p1)
        # p2Out = tissue.scan(p2)
