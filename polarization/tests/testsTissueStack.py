import envtest
from polarization import *

np.random.seed(521)


class TestTissueStack(envtest.MyTestCase):
    def setUp(self) -> None:
        self.k = 2 * np.pi / 1.3
        self.pIn = JonesVector.horizontal()
        self.pIn.k = self.k

    def testPropagateNoBirefringence(self):
        """ Should equal propagation in Vacuum. """
        stack = TissueStackNoBirefringence()
        stackLength = stack.offset + sum([layer.thickness for layer in stack.layers])

        pOut = stack.propagateThrough(self.pIn)

        self.assertAlmostEqual(pOut.Ex, exp(1j * self.pIn.k * stackLength))
        self.assertAlmostEqual(pOut.Ey, 0)

    def testPropagateWithOrientedTissueHorizontal(self):
        """ With all layer optic axes oriented with the beam (no Q/U components), there should be no retarding effect
        other than Vacuum propagation. """
        stack = TissueStackOrientedHorizontally()
        stackLength = stack.offset + sum([layer.thickness for layer in stack.layers])

        pOut = stack.propagateThrough(self.pIn)

        self.assertAlmostEqual(pOut.Ex, exp(1j * self.pIn.k * stackLength))
        self.assertAlmostEqual(pOut.Ey, 0)

    def testPropagateWithOrientedTissue45(self):
        """ With all layer optic axes oriented with the beam (no Q/U components), there should be no retarding effect
        other than Vacuum propagation. """
        self.pIn = JonesVector.plus45()
        self.pIn.k = self.k
        stack = TissueStackOriented45Degrees()
        stackLength = stack.offset + sum([layer.thickness for layer in stack.layers])

        pOut = stack.propagateThrough(self.pIn)

        self.assertAlmostEqual(pOut.Ex, sin(np.pi/4) * exp(1j * self.pIn.k * stackLength))
        self.assertAlmostEqual(pOut.Ey, sin(np.pi/4) * exp(1j * self.pIn.k * stackLength))

    def testPropagatePerpendicularLayer(self):
        """ With a beam polarization perpendicular to the one and only tissue optic axis,
        the retarding effect is simply ikL(1+dn). """
        self.pIn = JonesVector.vertical()
        self.pIn.k = self.k
        stack = TissueStackSingleHorizontal()
        stackLength = stack.offset + sum([layer.thickness for layer in stack.layers])
        layerLength = stack.layers[0].thickness
        layerBirefringence = stack.layers[0].birefringence

        pOut = stack.propagateThrough(self.pIn)

        self.assertAlmostEqual(pOut.Ex, 0)
        self.assertAlmostEqual(pOut.Ey, exp(1j * self.pIn.k * (stackLength + layerLength * layerBirefringence)))

    def testPropagateArbitraryLayer(self):
        stack = TissueStackSingle()
        layerLength = stack.layers[0].thickness
        layerBirefringence = stack.layers[0].birefringence

        jPhi = 1j * layerLength * self.pIn.k
        A = exp(jPhi)
        D = exp(jPhi * (1 + layerBirefringence))
        orientation = stack.layers[0].orientation
        s2 = sin(orientation)**2
        c2 = cos(orientation)**2
        cs = sin(orientation) * cos(orientation)
        Ap = c2 * A + D * s2
        Bp = A * cs - cs * D
        Dp = c2 * D + A * s2

        pOut = stack.propagateThrough(self.pIn)

        self.assertAlmostEqual(pOut.Ex, Ap * exp(1j * self.pIn.k * stack.offset))
        self.assertAlmostEqual(pOut.Ey, Bp * exp(1j * self.pIn.k * stack.offset))


    def testPropagateMany(self):
        stack = TissueStackUnit()
        res = 5
        pIn = Pulse.horizontal(centerWavelength=1.3, wavelengthBandwidth=0.13, resolution=res)
        pOut = stack.propagateManyThrough(pIn)

        self.assertTrue(len(pOut) == res)
        self.assertTrue(pOut[0].orientation != pOut[res//2].orientation)

    def testBackscatter(self):
        stack = TissueStackUnit()
        pOut = stack.backscatter(self.pIn)

        # print(pOut.orientation)
        # no ref to compare with

    def testBackscatterMany(self):
        stack = TissueStackUnit()
        res = 5
        pIn = Pulse.horizontal(centerWavelength=1.3, wavelengthBandwidth=0.13, resolution=5)

        pOut = stack.backscatterMany(pIn)

        self.assertTrue(len(pOut) == res)
        self.assertTrue(pOut[0].orientation != pOut[res//2].orientation)


class TissueStackUnit(TissueStack):
    def __init__(self):
        layers = [TissueLayer(birefringence=0.001, opticAxis=(1, 0, 0), scattDensity=20, thickness=400),
                  TissueLayer(birefringence=0.002, opticAxis=(1, 1, 0), scattDensity=10, thickness=600),
                  TissueLayer(birefringence=0.003, opticAxis=(0, 1, 0), scattDensity=20, thickness=800)]
        super(TissueStackUnit, self).__init__(offset=100, layers=layers)


class TissueStackNoBirefringence(TissueStack):
    def __init__(self):
        layers = [TissueLayer(birefringence=0, opticAxis=(1, 0, 0), scattDensity=20, thickness=400),
                  TissueLayer(birefringence=0, opticAxis=(1, 1, 0), scattDensity=10, thickness=600),
                  TissueLayer(birefringence=0, opticAxis=(0, 1, 0), scattDensity=20, thickness=800)]
        super(TissueStackNoBirefringence, self).__init__(offset=100, layers=layers)


class TissueStackOrientedHorizontally(TissueStack):
    def __init__(self):
        layers = [TissueLayer(birefringence=0.001, opticAxis=(1, 0, 0), scattDensity=20, thickness=400),
                  TissueLayer(birefringence=0.002, opticAxis=(1, 0, 0), scattDensity=10, thickness=600),
                  TissueLayer(birefringence=0.003, opticAxis=(1, 0, 0), scattDensity=20, thickness=800)]
        super(TissueStackOrientedHorizontally, self).__init__(offset=100, layers=layers)


class TissueStackOriented45Degrees(TissueStack):
    def __init__(self):
        layers = [TissueLayer(birefringence=0.001, opticAxis=(0, 1, 0), scattDensity=20, thickness=400),
                  TissueLayer(birefringence=0.002, opticAxis=(0, 1, 0), scattDensity=10, thickness=600),
                  TissueLayer(birefringence=0.003, opticAxis=(0, 1, 0), scattDensity=20, thickness=800)]
        super(TissueStackOriented45Degrees, self).__init__(offset=100, layers=layers)


class TissueStackSingleHorizontal(TissueStack):
    def __init__(self):
        layers = [TissueLayer(birefringence=0.001, opticAxis=(1, 0, 0), scattDensity=20, thickness=400)]
        super(TissueStackSingleHorizontal, self).__init__(offset=100, layers=layers)


class TissueStackSingle(TissueStack):
    def __init__(self):
        layers = [TissueLayer(birefringence=0.001, opticAxis=(0.6, 1.1, 0), scattDensity=20, thickness=400)]
        super(TissueStackSingle, self).__init__(offset=100, layers=layers)


if __name__ == '__main__':
    envtest.main()
