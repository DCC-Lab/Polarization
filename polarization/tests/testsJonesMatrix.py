import envtest
import unittest
from polarization.jonesmatrix import *
from polarization.jonesvector import JonesVector
from numpy import exp, pi, angle

class TestLayer(envtest.MyTestCase):
    def testDefaultInitJonesMatrix(self):
        m = JonesMatrix()
        
        self.assertIsNotNone(m)
        self.assertEqual(m.determinant,1)

    def testInitJonesMatrix(self):
        m = JonesMatrix(1,0,0,1,physicalLength=1.0)
        
        self.assertIsNotNone(m)
        self.assertEqual(m.determinant,1)
        self.assertEqual(m.L, 1)

    def testMultiplyJonesMatrix(self):
        m1 = JonesMatrix(1,0,0,1,physicalLength=1.0)
        m2 = JonesMatrix(1,0,0,1,physicalLength=2.0)
        
        m = m1*m2
        
        self.assertIsNotNone(m)
        self.assertEqual(m.determinant,1)
        self.assertEqual(m.L, 3)

    def testTransformJonesVector(self):
        m = JonesMatrix(1,0,0,1,physicalLength=1.0)
        v = JonesVector(1,0)
        self.assertEqual(v.z, 0)

        vOut = m*v

        self.assertIsNotNone(vOut)
        self.assertEqual(vOut.Ex, 1)
        self.assertEqual(vOut.Ey, 0)
        self.assertEqual(vOut.z, 1.0)

    def testHorizontalPolarizer(self):
        v = JonesVector(-1,1)
        m = HorizontalPolarizer()

        vOut = m*v

        self.assertEqual(vOut.Ex, v.Ex)
        self.assertEqual(vOut.Ey, 0)

    def testHorizontalPolarizerBlock(self):
        v = JonesVector(0,1)
        m = HorizontalPolarizer()

        vOut = m*v

        self.assertEqual(vOut.Ex, v.Ex)
        self.assertEqual(vOut.Ey, 0)

    def testVerticalPolarizer(self):
        v = JonesVector(-1,1)
        m = VerticalPolarizer()

        vOut = m*v

        self.assertEqual(vOut.Ex, 0)
        self.assertEqual(vOut.Ey, v.Ey)

    def testVerticalPolarizerBlock(self):
        v = JonesVector(-1,0)
        m = VerticalPolarizer()

        vOut = m*v

        self.assertEqual(vOut.Ex, 0)
        self.assertEqual(vOut.Ey, v.Ey)

    def testPlus45Polarizer(self):
        v = JonesVector(1,1)
        m = Plus45Polarizer()

        vOut = m*v

        self.assertEqual(vOut.Ex, v.Ex)
        self.assertEqual(vOut.Ey, v.Ey)

    def testPlus45PolarizerBlock(self):
        v = JonesVector(-1,1)
        m = Plus45Polarizer()

        vOut = m*v

        self.assertEqual(vOut.Ex, 0)
        self.assertEqual(vOut.Ey, 0)

    def testMinus45Polarizer(self):
        v = JonesVector(-1,1)
        m = Minus45Polarizer()

        vOut = m*v

        self.assertEqual(vOut.Ex, v.Ex)
        self.assertEqual(vOut.Ey, v.Ey)

    def testMinus45PolarizerBlock(self):
        v = JonesVector(1,1)
        m = Minus45Polarizer()

        vOut = m*v

        self.assertEqual(vOut.Ex, 0)
        self.assertEqual(vOut.Ey, 0)

    def testRightCircularPolarizer(self):
        v = JonesVector(1,1)
        m = RightCircularPolarizer()

        vOut = m*v

        self.assertTrue(vOut.isCircularlyPolarized)
        self.assertTrue(vOut.isRightCircularlyPolarized)
        self.assertFalse(vOut.isLeftCircularlyPolarized)

    def testLeftCircularPolarizer(self):
        v = JonesVector(1,1)
        m = LeftCircularPolarizer()

        vOut = m*v

        self.assertTrue(vOut.isCircularlyPolarized)
        self.assertTrue(vOut.isLeftCircularlyPolarized)
        self.assertFalse(vOut.isRightCircularlyPolarized)

    def testRetarder(self):
        ret = PhaseRetarder(delta=1)
        self.assertAlmostEqual(np.exp(1j), ret.A, 3)
        self.assertAlmostEqual(0, ret.B, 3)
        self.assertAlmostEqual(0, ret.C, 3)
        self.assertAlmostEqual(1, ret.D, 3)

    def testRetarderAsQWP(self):
        ret = PhaseRetarder(delta=np.pi/2) # Ex is delayed by pi/2
        self.assertAlmostEqual(np.exp(1j*np.pi/2), ret.A, 3)
        self.assertAlmostEqual(0, ret.B, 3)
        self.assertAlmostEqual(0, ret.C, 3)
        self.assertAlmostEqual(1, ret.D, 3)
        
        v = JonesVector(1,1)
        vOut = ret*v

        self.assertTrue(vOut.isCircularlyPolarized)
        self.assertTrue(vOut.isRightCircularlyPolarized)
        self.assertFalse(vOut.isLeftCircularlyPolarized)

    def testQuarterWaveplateInit(self):
        qwp = QWP(theta=0)
        ret = PhaseRetarder(delta=-np.pi/2) # x is fast axis

        self.assertAlmostEqual(qwp.A, ret.A, 3)
        self.assertAlmostEqual(qwp.B, ret.B, 3)
        self.assertAlmostEqual(qwp.C, ret.C, 3)
        self.assertAlmostEqual(qwp.D, ret.D, 3)

    def testQuarterWaveplate(self):
        v = JonesVector(1,1)
        m = QWP(theta=0) # Ex is ahead = left circular

        vOut = m*v

        self.assertTrue(vOut.isCircularlyPolarized)
        self.assertTrue(vOut.isLeftCircularlyPolarized)
        self.assertFalse(vOut.isRightCircularlyPolarized)

        v = JonesVector(1,1)
        m = QWP(theta=np.pi/2)

        vOut = m*v

        self.assertTrue(vOut.isCircularlyPolarized)
        self.assertTrue(vOut.isRightCircularlyPolarized)
        self.assertFalse(vOut.isLeftCircularlyPolarized)

    def testHalfWaveplate(self):
        v = JonesVector(1,1)
        m = HWP(theta=0) # Ex is ahead by pi

        vOut = m*v

        self.assertAlmostEqual(vOut.Ex.real, -1, 5)
        self.assertAlmostEqual(vOut.Ey.real, 1, 5)

        v = JonesVector(1,1)
        m = HWP(theta=np.pi/2) # Ey is ahead by pi

        vOut = m*v

        self.assertAlmostEqual(vOut.Ex.real, 1, 5)
        self.assertAlmostEqual(vOut.Ey.real, -1, 5)

        v = JonesVector(1,0)
        m = HWP(theta=np.pi/4) # +45 is ahead by pi

        vOut = m*v

        self.assertAlmostEqual(vOut.Ex.real, 0, 5)
        self.assertAlmostEqual(vOut.Ey.real, -1, 5)

    def testRotation(self):
        v = Rotation(theta=np.pi/2)*JonesVector(1,0)
        self.assertAlmostEqual(abs(v.Ex), 0, 5)
        self.assertAlmostEqual(abs(v.Ey), 1, 5)

    # def testInitJonesMatrixFromOpticalProperties(self):
    #     v = JonesM
    #     JonesMatrix(1, 0, 0, 1, physicalLength=1.0,)
    #     pass


if __name__ == '__main__':
    unittest.main()
