import envtest
import unittest
from polarization.jonesvector import *
from polarization.jonesmatrix import *
from numpy import exp, pi, angle
import matplotlib
import matplotlib.pyplot as plt

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

    def testIntensity(self):
        v = JonesVector(1, 1)
        self.assertEqual(v.intensity,2)

    def testStokesComponentS0S1S2(self):
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

    def testNormalization(self):
        v = JonesVector(1, 1)
        v.normalize()
        self.assertAlmostEqual(v.intensity, 1.0, 5)

    def testStokesComponentS0S1S2AreReal(self):
        v = JonesVector(1, 0)
        self.assertFalse(isinstance(v.S0, complex))
        self.assertFalse(isinstance(v.S1, complex))
        self.assertFalse(isinstance(v.S2, complex))

        self.assertTrue(isinstance(v.S0, float))
        self.assertTrue(isinstance(v.S1, float))
        self.assertTrue(isinstance(v.S2, float))

        with self.assertRaises(NotImplementedError):
            v.S3
        with self.assertRaises(NotImplementedError):
            v.StokesVector


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

    # def testAnimate(self):
    #     v = JonesVector(Ex=exp(1j*pi/3), Ey=0.3).normalize()
    #     v = QWP(theta=pi/5)*v
    #     v.show()

    #     v = JonesVector(Ex=1, Ey=0).normalize()
    #     v.show()

    def testPredefined(self):
        v = JonesVector.rightCircular()
        self.assertIsNotNone(v)
        self.assertTrue(v.isRightCircularlyPolarized)

        v = JonesVector.leftCircular()
        self.assertIsNotNone(v)
        self.assertTrue(v.isLeftCircularlyPolarized)

        v = JonesVector.horizontal()
        self.assertIsNotNone(v)
        self.assertTrue(v.isLinearlyPolarized)

        v = JonesVector.vertical()
        self.assertIsNotNone(v)
        self.assertTrue(v.isLinearlyPolarized)

        v = JonesVector.plus45()
        self.assertIsNotNone(v)
        self.assertTrue(v.isLinearlyPolarized)

        v = JonesVector.minus45()
        self.assertIsNotNone(v)
        self.assertTrue(v.isLinearlyPolarized)

    def testOrientation(self):
        self.assertAlmostEqual(JonesVector.vertical().orientation*degPerRad, 90)
        self.assertAlmostEqual(JonesVector.horizontal().orientation*degPerRad, 0)
        self.assertAlmostEqual(JonesVector.plus45().orientation*degPerRad, 45)
        self.assertAlmostEqual(JonesVector.minus45().orientation*degPerRad, -45)
        self.assertAlmostEqual(JonesVector.rightCircular().orientation*degPerRad, 45)
        self.assertAlmostEqual(JonesVector.leftCircular().orientation*degPerRad, 45)

if __name__ == '__main__':
    unittest.main()
