import envtest  # modifies path
from polarization import JonesVector
import matplotlib.pyplot as plt
import numpy as np


class Beam:
    def __init__(self, vectors=None):
        """
        A list of JonesVector to define a pulse with a uniform intensity distribution.
        Provide 'vectors' if they are already defined, or 'centerWavelength', 'wavelengthBandwidth' and 'polarization'
         to create a new pulse with a given polarization. This pulse will have as much JonesVectors as 'resolution'.
        """
        self.vectors = []
        if vectors is not None:
            self.vectors = vectors

    @property
    def k(self) -> list:
        return [v.k for v in self.vectors]

    @property
    def kc(self) -> float:
        return np.mean(self.k)

    @property
    def Ex(self):
        return [v.Ex for v in self.vectors]

    @property
    def Ey(self):
        return [v.Ey for v in self.vectors]

    def display(self):
        # todo: display fringes
        pass

    def __iter__(self):
        return iter(self.vectors)

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, item):
        return self.vectors[item]


class TestBeam(envtest.MyTestCase):
    def testInit(self):
        self.assertTrue(True)

    def testBeamInit(self):
        self.assertIsNotNone(Beam())

    def testBeam(self):
        self.assertIsNotNone(Beam([JonesVector.vertical()]))

if __name__ == '__main__':
    unittest.main()
