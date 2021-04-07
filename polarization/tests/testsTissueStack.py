import envtest
from polarization.tissueStack import *
from polarization.pulse import *

np.random.seed(521)


class TestTissueStack(envtest.MyTestCase):
    def setUp(self) -> None:
        self.stack = TissueStackTestUnit()

        self.k = 2 * np.pi / 1.3
        self.pIn = JonesVector.horizontal()
        self.pIn.k = self.k


class TissueStackTestUnit(TissueStack):
    def __init__(self):
        layers = [TissueLayer(birefringence=0.001, opticAxis=(1, 0, 0), scattDensity=20, thickness=400),
                  TissueLayer(birefringence=0.002, opticAxis=(1, 1, 0), scattDensity=10, thickness=600),
                  TissueLayer(birefringence=0.003, opticAxis=(0, 1, 0), scattDensity=20, thickness=800)]
        super(TissueStackTestUnit, self).__init__(offset=100, layers=layers)


if __name__ == '__main__':
    envtest.main()
