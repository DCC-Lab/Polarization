import envtest
from polarization.tissue import *

np.random.seed(521)


class TestTissueLayer(envtest.MyTestCase):
    def setUp(self) -> None:
        pass

    def testRandomTissue2DInit(self):
        tissue = RandomTissue2D()
        unique, counts = np.unique(tissue.map[0], return_counts=True, axis=0)
        print(unique, counts)
