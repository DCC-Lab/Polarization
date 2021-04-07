import envtest
from polarization.tissue import *

np.random.seed(521)


class TestTissueLayer(envtest.MyTestCase):
    def setUp(self) -> None:
        pass

    def testRandomTissue2DMap(self):
        tissue = RandomTissue2D(nLayers=6)
        self.assertEqual(tissue.map.shape, (7, 200))

    def testTissueStackAt(self):
        tissue = RandomTissue2D(nLayers=6)

        stack = tissue.stackAt(5)
        self.assertAlmostEqual(len(stack), 6)
