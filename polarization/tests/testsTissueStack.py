import envtest
from polarization import *
from polarization.tissueLayer import Scatterers

np.random.seed(521)


class TestTissuePropagation(envtest.MyTestCase):
    def __init__(self, *args, **kwargs):
        self.k = None
        self.pIn = None
        super(TestTissuePropagation, self).__init__(*args, **kwargs)

    def setUp(self):
        self.k = 2 * np.pi / 1.3
        self.pIn = JonesVector.horizontal()
        self.pIn.k = self.k

    def testPropagateThroughNoBirefringence(self):
        """ Should equal propagation in Vacuum. """
        stack = TissueStackNoBirefringence()
        stackLength = stack.offset + sum([layer.thickness for layer in stack.layers])

        pOut = stack.propagateThrough(self.pIn)

        self.assertAlmostEqual(pOut.Ex, exp(1j * self.pIn.k * stackLength))
        self.assertAlmostEqual(pOut.Ey, 0)

    def testPropagateThroughOrientedTissueHorizontal(self):
        """ With all layer optic axes oriented with the beam (no Q/U components), there should be no retarding effect
        other than Vacuum propagation. """
        stack = TissueStackOrientedHorizontally()
        stackLength = stack.offset + sum([layer.thickness for layer in stack.layers])

        pOut = stack.propagateThrough(self.pIn)

        self.assertAlmostEqual(pOut.Ex, exp(1j * self.pIn.k * stackLength))
        self.assertAlmostEqual(pOut.Ey, 0)

    def testPropagateThroughOrientedTissue45(self):
        """ With all layer optic axes oriented with the beam (no Q/U components), there should be no retarding effect
        other than Vacuum propagation. """
        self.pIn = JonesVector.plus45()
        self.pIn.k = self.k
        stack = TissueStackOriented45Degrees()
        stackLength = stack.offset + sum([layer.thickness for layer in stack.layers])

        pOut = stack.propagateThrough(self.pIn)

        self.assertAlmostEqual(pOut.Ex, sin(np.pi / 4) * exp(1j * self.pIn.k * stackLength))
        self.assertAlmostEqual(pOut.Ey, sin(np.pi / 4) * exp(1j * self.pIn.k * stackLength))

    def testPropagateThroughPerpendicularLayer(self):
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

    def testPropagateThroughSingleLayer(self):
        """ With a single arbitrary birefringent layer, the test becomes more involved.
        For a single rotation matrix on vacuum propagation, we have [[A, B], [C, D]] * [e^ikl, 0]^T
        """
        stack = TissueStackSingle()
        layerLength = stack.layers[0].thickness
        layerBirefringence = stack.layers[0].birefringence

        jPhi = 1j * layerLength * self.pIn.k
        A = exp(jPhi)
        D = exp(jPhi * (1 + layerBirefringence))
        orientation = stack.layers[0].orientation
        Ap = cos(orientation) ** 2 * A + D * sin(orientation) ** 2
        Bp = (A - D) * sin(orientation) * cos(orientation)
        Dp = cos(orientation) ** 2 * D + A * sin(orientation) ** 2

        pOut = stack.propagateThrough(self.pIn)

        self.assertAlmostEqual(pOut.Ex, Ap * exp(1j * self.pIn.k * stack.offset))
        self.assertAlmostEqual(pOut.Ey, Bp * exp(1j * self.pIn.k * stack.offset))

    def testPropagateThrough(self):
        """ With multiple arbitrary birefringent layers, we have to use the results from last layer
        and since the input at secondary layers is not horizontal, the math is a little more complex.
        """
        # todo, maybe with numpy.matmul ...
        # fisrt layer : [[A, B], [C, D]] * [e^ikl, 0]^T = e^ikL * [A , B]^T
        # second layer: [[E, F], [G, H]] * e^ikL * [A, B]^T = ...
        pass

    def testPropagateManyThrough(self):
        """ Qualitative test for pulse propagation (multiple vectors). """
        # todo: quantitative test

        stack = TissueStackUnit()
        res = 5
        pIn = Pulse.horizontal(centerWavelength=1.3, wavelengthBandwidth=0.13, resolution=res)
        pOut = stack.propagateManyThrough(pIn)

        self.assertTrue(len(pOut) == res)
        self.assertTrue(pOut[0].orientation != pOut[res // 2].orientation)


class TestTissueBackscattering(envtest.MyTestCase):
    def __init__(self, *args, **kwargs):
        self.k = None
        self.pIn = None
        super(TestTissueBackscattering, self).__init__(*args, **kwargs)

    def setUp(self):
        self.k = 2 * np.pi / 1.3
        self.pIn = JonesVector.horizontal()
        self.pIn.k = self.k

    def testBackscatterSingleScattererSingleLayerNoBirefringence(self):
        stack = TissueStackSingleScattererSingleLayerNoBirefringence()

        scatPosition = stack.offset + stack.layers[0].scatterers.dz[0]
        scatStrength = stack.layers[0].scatterers.strength[0]

        pOut = stack.backscatter(self.pIn)

        self.assertAlmostEqual(pOut.Ex, scatStrength * exp(1j * self.pIn.k * 2 * scatPosition))
        self.assertAlmostEqual(pOut.Ey, 0)

    def testBackscatterSingleLayerNoBirefringence(self):
        stack = TissueStackSingleLayerNoBirefringence()
        expectedEx = 0

        scatterers = stack.layers[0].scatterers
        for (dz, strength) in zip(scatterers.dz, scatterers.strength):
            scatPosition = stack.offset + dz
            expectedEx += strength * exp(1j * self.pIn.k * 2 * scatPosition)

        pOut = stack.backscatter(self.pIn)

        self.assertAlmostEqual(pOut.Ex, expectedEx)
        self.assertAlmostEqual(pOut.Ey, 0)

    def testBackscatterSingleLayerPerpendicular(self):
        """ With a beam polarization perpendicular to the one and only tissue optic axis, the retarding effect
         of each scatterer is simply ikL(1+dn). """
        self.pIn = JonesVector.vertical()
        self.pIn.k = self.k
        stack = TissueStackSingleHorizontal()
        layer = stack.layers[0]

        expectedEy = 0
        for (dz, strength) in zip(layer.scatterers.dz, layer.scatterers.strength):
            expectedEy += strength * exp(1j * self.pIn.k * 2 * (stack.offset + dz + dz * layer.birefringence))

        pOut = stack.backscatter(self.pIn)

        self.assertAlmostEqual(pOut.Ex, 0)
        self.assertAlmostEqual(pOut.Ey, expectedEy)

    def testBackscatterSingleLayer(self):
        stack = TissueStackSingle()
        layer = stack.layers[0]

        expectedEx = 0
        expectedEy = 0
        for (dz, strength) in zip(layer.scatterers.dz, layer.scatterers.strength):
            jPhi = 1j * self.pIn.k * 2 * dz
            A = exp(jPhi)
            D = exp(jPhi * (1 + layer.birefringence))
            a = layer.orientation
            Ap = cos(a) ** 2 * A + D * sin(a) ** 2
            Bp = (A - D) * sin(a) * cos(a)
            Dp = cos(a) ** 2 * D + A * sin(a) ** 2

            expectedEx += Ap * strength * exp(1j * self.pIn.k * 2 * stack.offset)
            expectedEy += Bp * strength * exp(1j * self.pIn.k * 2 * stack.offset)

        pOut = stack.backscatter(self.pIn)

        self.assertAlmostEqual(pOut.Ex, expectedEx)
        self.assertAlmostEqual(pOut.Ey, expectedEy)

    def testBackscatter(self):
        # combine testPropagateThrough with testBackscatterSingleLayer
        # todo
        pass

    def testBackscatterMany(self):
        """ Qualitative test for pulse backscatter (multiple vectors). """
        # todo: quantitative test

        stack = TissueStackUnit()
        res = 5
        pIn = Pulse.horizontal(centerWavelength=1.3, wavelengthBandwidth=0.13, resolution=5)

        pOut = stack.backscatterMany(pIn)

        self.assertTrue(len(pOut) == res)
        self.assertTrue(pOut[0].orientation != pOut[res // 2].orientation)


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
        layers = [TissueLayer(birefringence=0.001, opticAxis=(1, 0, 0), scattDensity=0.05, thickness=400)]
        super(TissueStackSingleHorizontal, self).__init__(offset=100, layers=layers)


class TissueStackSingle(TissueStack):
    def __init__(self):
        layers = [TissueLayer(birefringence=0.001, opticAxis=(0.6, 1.1, 0), scattDensity=20, thickness=400)]
        super(TissueStackSingle, self).__init__(offset=100, layers=layers)


class TissueStackSingleScattererSingleLayerNoBirefringence(TissueStack):
    def __init__(self):
        layers = [TissueLayer(birefringence=0, opticAxis=(1, 0, 0), scattDensity=20, thickness=400)]
        for layer in layers:
            layer.scatterers = Scatterers(layer.thickness, N=1)
        super(TissueStackSingleScattererSingleLayerNoBirefringence, self).__init__(offset=100, layers=layers)


class TissueStackSingleLayerNoBirefringence(TissueStack):
    def __init__(self):
        layers = [TissueLayer(birefringence=0, opticAxis=(1, 0, 0), scattDensity=0.05, thickness=400)]
        super(TissueStackSingleLayerNoBirefringence, self).__init__(offset=100, layers=layers)


if __name__ == '__main__':
    envtest.main()
