import envtest
import unittest
from polarization.jonesmatrix import *
from polarization.jonesvector import JonesVector
from numpy import exp, pi, angle, array, matmul
from numpy.linalg import eig, eigh

class TestMatrices(envtest.MyTestCase):
    def testArrayRowCol(self):
        m = array([[1,2],[3,4]])
        self.assertIsNotNone(m)
        self.assertEqual(m[0,0], 1)
        self.assertEqual(m[0,1], 2)
        self.assertEqual(m[1,0], 3)
        self.assertEqual(m[1,1], 4)

    def testMatrixProduct(self):
        m = array([[1,2],[3,4]])
        v = array([5,6])
        r = matmul(m,v)
        self.assertEqual(r[0], 17)
        self.assertEqual(r[1], 39)

    def testDefaultInitJonesMatrix(self):
        m = JonesMatrix()
        
        self.assertIsNotNone(m)
        self.assertEqual(m.determinant,1)

    def testDynamicPropertiesJonesMatrix(self):
        m = JonesMatrix()
        
        m.setValue('L', 2)        
        self.assertEqual(m.L,2)
        self.assertEqual(m.value('L'),2)

    def testInitJonesMatrix(self):
        m = JonesMatrix(1,0,0,1,physicalLength=1.0)
        
        self.assertIsNotNone(m)
        self.assertEqual(m.determinant,1)
        self.assertEqual(m.L, 1)

    def testInitJonesMatrixABCD(self):
        m = JonesMatrix(1,2,3,4,physicalLength=1.0)
        
        self.assertIsNotNone(m)
        self.assertEqual(m.L, 1)
        self.assertEqual(m.A, 1)
        self.assertEqual(m.B, 2)
        self.assertEqual(m.C, 3)
        self.assertEqual(m.D, 4)

    def testMultiplyJonesMatrix(self):
        m1 = JonesMatrix(1,0,0,1,physicalLength=1.0)
        m2 = JonesMatrix(1,0,0,1,physicalLength=2.0)
        
        m = m1*m2
        
        self.assertIsNotNone(m)
        self.assertEqual(m.determinant,1)
        self.assertEqual(m.L, 3)

    def testTransformJonesVector(self):
        m = JonesMatrix(1,2,3,4,physicalLength=1.0)
        v = JonesVector(5,6)
        self.assertEqual(v.z, 0)

        vOut = m*v

        self.assertIsNotNone(vOut)
        self.assertEqual(vOut.Ex, 17)
        self.assertEqual(vOut.Ey, 39)
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

    def testHorizontalPolarizerRotatedBy90GivesVertical(self):
        h = HorizontalPolarizer().rotatedBy(pi/2)
        v = VerticalPolarizer()

        self.assertAlmostEqual(h.A,v.A)
        self.assertAlmostEqual(h.B,v.B)
        self.assertAlmostEqual(h.C,v.C)
        self.assertAlmostEqual(h.D,v.D)

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
        self.assertAlmostEqual(exp(1j), ret.A, 3)
        self.assertAlmostEqual(0, ret.B, 3)
        self.assertAlmostEqual(0, ret.C, 3)
        self.assertAlmostEqual(1, ret.D, 3)

    def testRetarderAsQWP(self):
        ret = PhaseRetarder(delta=pi/2) # Ex is delayed by pi/2
        self.assertAlmostEqual(exp(1j*pi/2), ret.A, 3)
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
        ret = PhaseRetarder(delta=-pi/2) # x is fast axis

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
        m = QWP(theta=pi/2)

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
        m = HWP(theta=pi/2) # Ey is ahead by pi

        vOut = m*v

        self.assertAlmostEqual(vOut.Ex.real, 1, 5)
        self.assertAlmostEqual(vOut.Ey.real, -1, 5)

        v = JonesVector(1,0)
        m = HWP(theta=pi/4) # +45 is ahead by pi/4

        vOut = m*v

        self.assertAlmostEqual(vOut.Ex.real, 0, 5)
        self.assertAlmostEqual(vOut.Ey.real, -1, 5)

    def testRotation(self):
        v = Rotation(theta=pi/2)*JonesVector(1,0)
        self.assertAlmostEqual(abs(v.Ex), 0, 5)
        self.assertAlmostEqual(abs(v.Ey), 1, 5)

    # def testInitJonesMatrixFromOpticalProperties(self):
    #     v = JonesM
    #     JonesMatrix(1, 0, 0, 1, physicalLength=1.0,)
    #     pass
    def testEigens(self):
        m = QWP(theta=pi/2)
        w, v = eig(m.asArray)

        self.assertAlmostEqual(abs(w[0]), abs(w[1]), 6)
        self.assertAlmostEqual(angle(w[0]) - angle(w[1]), pi/2, 6)

    def testBirefringenceInWaveplates(self):
        self.assertTrue(QWP(theta=0).isBirefringent)
        self.assertTrue(QWP(theta=45*radPerDeg).isBirefringent)
        self.assertTrue(QWP(theta=90*radPerDeg).isBirefringent)
        self.assertTrue(QWP(theta=180*radPerDeg).isBirefringent)

        self.assertTrue(HWP(theta=0).isBirefringent)
        self.assertTrue(HWP(theta=45*radPerDeg).isBirefringent)
        self.assertTrue(HWP(theta=90*radPerDeg).isBirefringent)
        self.assertTrue(HWP(theta=180*radPerDeg).isBirefringent)

    def testNoBirefringenceInOtherMatrices(self):
        self.assertFalse(HorizontalPolarizer().isBirefringent)
        self.assertFalse(VerticalPolarizer().isBirefringent)
        self.assertFalse(Plus45Polarizer().isBirefringent)
        self.assertFalse(Minus45Polarizer().isBirefringent)
        self.assertFalse(RightCircularPolarizer().isBirefringent)
        self.assertFalse(LeftCircularPolarizer().isBirefringent)

    def testRotationNotBirefringent(self):
        m = HorizontalPolarizer().rotatedBy(theta=pi/3)
        self.assertFalse(m.isBirefringent)

if __name__ == '__main__':
    unittest.main()
