import envtest
from polarization import *

np.random.seed(521)


class TestTissueStack(envtest.MyTestCase):
    def setUp(self) -> None:
        self.stack = TissueStackTestUnit()

        self.k = 2 * np.pi / 1.3
        self.pIn = JonesVector.horizontal()
        self.pIn.k = self.k

    def testPropagate(self):
        """ When using our not phase-symmetric matrix, we expect same output orientation, but different phase."""
        pOut = self.stack.propagateThrough(self.pIn)

        # print(pOut.orientation)
        # no ref to compare with

    def testPropagateMany(self):
        res = 5
        pIn = Pulse.horizontal(centerWavelength=1.3, wavelengthBandwidth=0.13, resolution=res)
        pOut = self.stack.propagateManyThrough(pIn)

        self.assertTrue(len(pOut) == res)
        self.assertTrue(pOut[0].orientation != pOut[res//2].orientation)

    def testBackscatter(self):
        pOut = self.stack.backscatter(self.pIn)

        # print(pOut.orientation)
        # no ref to compare with

    def testBackscatterMany(self):
        res = 5
        pIn = Pulse.horizontal(centerWavelength=1.3, wavelengthBandwidth=0.13, resolution=5)

        pOut = self.stack.backscatterMany(pIn)

        self.assertTrue(len(pOut) == res)
        self.assertTrue(pOut[0].orientation != pOut[res//2].orientation)


class TissueStackTestUnit(TissueStack):
    def __init__(self):
        layers = [TissueLayer(birefringence=0.001, opticAxis=(1, 0, 0), scattDensity=20, thickness=400),
                  TissueLayer(birefringence=0.002, opticAxis=(1, 1, 0), scattDensity=10, thickness=600),
                  TissueLayer(birefringence=0.003, opticAxis=(0, 1, 0), scattDensity=20, thickness=800)]
        super(TissueStackTestUnit, self).__init__(offset=100, layers=layers)


if __name__ == '__main__':
    envtest.main()
