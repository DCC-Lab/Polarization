import envtest
import unittest
from polarization import *
from numpy import exp, pi, angle

class TestVector(envtest.MyTestCase):
    def testInitJones(self):
        v = JonesVector()
        self.assertIsNotNone(v)

        v = JonesVector(1,0)
        self.assertIsNotNone(v)

        v = JonesVector(complex(1,0),0)
        self.assertIsNotNone(v)

        v = JonesVector(0, complex(1,0))
        self.assertIsNotNone(v)

        v = JonesVector(complex(1,0), complex(1,0))
        self.assertIsNotNone(v)
        self.assertEqual(v.z,0)

    def testComplexFunctions(self):
        n = complex(exp(-1j))
        self.assertEqual(angle(n), -1)
        n = complex(exp(10j))
        self.assertAlmostEqual(angle(n), 10 - pi*4,3)


    def testComponentsAreComplex(self):
        v = JonesVector(complex(1,0), complex(1,0))
        self.assertTrue(isinstance(v.Ex, complex))
        self.assertTrue(isinstance(v.Ey, complex))

        v = JonesVector(1,1)
        self.assertTrue(isinstance(v.Ex, complex))
        self.assertTrue(isinstance(v.Ey, complex))

    def testStokesComponentsIntensity(self):
        v = JonesVector(1, 0)
        self.assertEqual(v.S0,1)

        v = JonesVector(0,1)
        self.assertEqual(v.S0,1)

        v = JonesVector(1,1)
        self.assertEqual(v.S0,2)

        v = JonesVector(exp(1j),exp(2j))
        self.assertEqual(v.S0,2)

        v = JonesVector(exp(1j),exp(-2j))
        self.assertEqual(v.S0,2)

        v = JonesVector(exp(-1j),exp(2j))
        self.assertEqual(v.S0,2)

        v = JonesVector(-exp(-1j),-exp(2j))
        self.assertEqual(v.S0,2)

    def testStokesComponentS1S2(self):
        v = JonesVector(1, 0)
        self.assertEqual(v.S0,1)
        self.assertEqual(v.S1,1)
        self.assertEqual(v.S2,0)

        v = JonesVector(0, 1)
        self.assertEqual(v.S0,1)
        self.assertEqual(v.S1,-1)
        self.assertEqual(v.S2,0)

        v = JonesVector(1, 1)
        self.assertEqual(v.S0,2)
        self.assertEqual(v.S1,0)
        self.assertEqual(v.S2,2)

        v = JonesVector(-1, -1)
        self.assertEqual(v.S0,2)
        self.assertEqual(v.S1,0)
        self.assertEqual(v.S2,2)

        v = JonesVector(1, -1)
        self.assertEqual(v.S0,2)
        self.assertEqual(v.S1,0)
        self.assertEqual(v.S2,-2)

        v = JonesVector(-1, 1)
        self.assertEqual(v.S0,2)
        self.assertEqual(v.S1,0)
        self.assertEqual(v.S2,-2)

    def testLinearPolarization(self):
        v = JonesVector(1, 0)
        self.assertTrue(v.isLinearlyPolarized)

        v = JonesVector(0, 1)
        self.assertTrue(v.isLinearlyPolarized)

        v = JonesVector(1, exp(-1j))
        self.assertFalse(v.isLinearlyPolarized)

        v = JonesVector(exp(-1j), exp(-1j))
        self.assertTrue(v.isLinearlyPolarized)

    def testCircularPolarization(self):
        v = JonesVector(1, 0)
        self.assertFalse(v.isCircularlyPolarized)

        v = JonesVector(1, exp(-1j*pi/2))
        self.assertTrue(v.isRightCircularlyPolarized)

        v = JonesVector(1, exp(1j*pi/2))
        self.assertTrue(v.isLeftCircularlyPolarized)

        v = JonesVector(exp(1j*pi/2), 1)
        self.assertTrue(v.isRightCircularlyPolarized)

        v = JonesVector(exp(-1j*pi/2), 1)
        self.assertTrue(v.isLeftCircularlyPolarized)

        v = JonesVector(exp(-1j*pi/2), 0.9)
        self.assertFalse(v.isCircularlyPolarized)
        self.assertTrue(v.isEllipticallyPolarized)

    def testCircularPolarizationLargePhase(self):
        v = JonesVector(1, exp(-1j*5*pi/2))
        self.assertTrue(v.isRightCircularlyPolarized)

        v = JonesVector(1, exp(1j*5*pi/2))
        self.assertTrue(v.isLeftCircularlyPolarized)

        v = JonesVector(exp(1j*5*pi/2), 1)
        self.assertTrue(v.isRightCircularlyPolarized)

        v = JonesVector(exp(-1j*5*pi/2), 1)
        self.assertTrue(v.isLeftCircularlyPolarized)

        v = JonesVector(exp(-1j*5*pi/2), 0.9)
        self.assertFalse(v.isCircularlyPolarized)
        self.assertTrue(v.isEllipticallyPolarized)


if __name__ == '__main__':
    unittest.main()
