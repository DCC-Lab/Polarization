import envtest
import unittest
from polarization.jonesmatrix import *
from polarization.jonesvector import *
from numpy import exp, pi, angle, array, matmul, arctan2
from numpy.linalg import eig, eigh


class TestMatrices(envtest.MyTestCase):
    def testArrayRowCol(self):
        m = array([[1, 2], [3, 4]])
        self.assertIsNotNone(m)
        self.assertEqual(m[0, 0], 1)
        self.assertEqual(m[0, 1], 2)
        self.assertEqual(m[1, 0], 3)
        self.assertEqual(m[1, 1], 4)

    def testMatrixProduct(self):
        m = array([[1, 2], [3, 4]])
        v = array([5, 6])
        r = matmul(m, v)
        self.assertEqual(r[0], 17)
        self.assertEqual(r[1], 39)

    def testDefaultInitJonesMatrix(self):
        m = JonesMatrix()

        self.assertIsNotNone(m)
        with self.assertRaises(ValueError) as context:
            m.m
        with self.assertRaises(ValueError) as context:
            m.m(k=1)
        with self.assertRaises(ValueError) as context:
            m.determinant

    def testDynamicPropertiesJonesMatrix(self):
        m = JonesMatrix()

        m.setValue('L', 2)
        self.assertEqual(m.L, 2)
        self.assertEqual(m.value('L'), 2)

    def testInitJonesMatrix(self):
        m = JonesMatrix(1, 0, 0, 1, physicalLength=1.0)

        self.assertIsNotNone(m)
        self.assertEqual(m.determinant, 1)
        self.assertEqual(m.L, 1)

    def testInitJonesMatrixABCD(self):
        m = JonesMatrix(1, 2, 3, 4, physicalLength=1.0)

        self.assertIsNotNone(m)
        self.assertEqual(m.L, 1)

        self.assertEqual(m.A, 1)
        self.assertEqual(m.B, 2)
        self.assertEqual(m.C, 3)
        self.assertEqual(m.D, 4)

    def testOrientedJonesMatrixABCD(self):
        m = JonesMatrix(1, 2, 3, 4, physicalLength=1.0)
        determinant = m.determinant
        for theta in range(0, 370, 10):
            m.orientation = theta * radPerDeg
            self.assertAlmostEqual(m.determinant, determinant)

    def testFlippedJonesMatrixABCD(self):
        m = JonesMatrix(1, 2, 3, 4, physicalLength=1.0)
        determinant = m.determinant
        m.orientation = 2 * pi
        self.assertAlmostEqual(m.determinant, determinant)
        self.assertAlmostEqual(m.A, 1)
        self.assertAlmostEqual(m.B, 2)
        self.assertAlmostEqual(m.C, 3)
        self.assertAlmostEqual(m.D, 4)

        m.orientation = pi
        self.assertAlmostEqual(m.determinant, determinant)
        self.assertAlmostEqual(m.A, 1)
        self.assertAlmostEqual(m.B, 2)
        self.assertAlmostEqual(m.C, 3)
        self.assertAlmostEqual(m.D, 4)

    def testPolarizerOrientation(self):
        h = HorizontalPolarizer()
        v = VerticalPolarizer()
        h.orientation = 0
        v.orientation = -pi / 2

        p = LinearPolarizer(theta=0)
        self.assertAlmostEqual(h.A, p.A)
        self.assertAlmostEqual(h.B, p.B)
        self.assertAlmostEqual(h.C, p.C)
        self.assertAlmostEqual(h.D, p.D)

        self.assertAlmostEqual(v.A, p.A)
        self.assertAlmostEqual(v.B, p.B)
        self.assertAlmostEqual(v.C, p.C)
        self.assertAlmostEqual(v.D, p.D)

        h.orientation = pi / 2
        v.orientation = 0

        p = LinearPolarizer(theta=pi / 2)
        self.assertAlmostEqual(h.A, p.A)
        self.assertAlmostEqual(h.B, p.B)
        self.assertAlmostEqual(h.C, p.C)
        self.assertAlmostEqual(h.D, p.D)

        self.assertAlmostEqual(v.A, p.A)
        self.assertAlmostEqual(v.B, p.B)
        self.assertAlmostEqual(v.C, p.C)
        self.assertAlmostEqual(v.D, p.D)

    def testMultiplyJonesMatrix(self):
        m1 = JonesMatrix(1, 0, 0, 1, physicalLength=1.0)
        m2 = JonesMatrix(1, 0, 0, 1, physicalLength=2.0)

        m = m1 * m2

        self.assertIsNotNone(m)
        self.assertEqual(m.determinant, 1)
        self.assertEqual(m.L, 3)

    def testMultiplyJonesVectorForward(self):
        m = JonesMatrix(1, 2, 3, 4, physicalLength=1.0)
        v = JonesVector(5, 6)
        self.assertEqual(v.z, 0)

        vOut = m * v

        self.assertIsNotNone(vOut)
        self.assertEqual(vOut.z, 1.0)

    def testMultiplyJonesVectorBackward(self):
        matrix = JonesMatrix(1, 2, 3, 4, physicalLength=1.0)
        v = JonesVector(5, 6)
        self.assertEqual(v.z, 0)

        m = matrix.backward()
        v.reflect()

        vOut = m * v

        self.assertIsNotNone(vOut)
        self.assertEqual(vOut.z, -1.0)

    def testTransformJonesVectorNoK(self):
        m = JonesMatrix(1, 2, 3, 4, physicalLength=1.0)
        v = JonesVector(5, 6)
        self.assertEqual(v.z, 0)

        vOut = m * v

        self.assertIsNotNone(vOut)
        self.assertEqual(vOut.Ex, 17)
        self.assertEqual(vOut.Ey, 39)
        self.assertEqual(vOut.z, 1.0)
        self.assertEqual(vOut.k, None)

    def testTransformJonesVectorWithK(self):
        m = JonesMatrix(1, 2, 3, 4, physicalLength=1.0)
        v = JonesVector(5, 6, k=6.28)
        self.assertEqual(v.z, 0)

        vOut = m * v

        self.assertIsNotNone(vOut)
        self.assertEqual(vOut.Ex, 17)
        self.assertEqual(vOut.Ey, 39)
        self.assertEqual(vOut.z, 1.0)
        self.assertEqual(vOut.k, 6.28)

    def testHorizontalPolarizer(self):
        v = JonesVector(-1, 1)
        m = HorizontalPolarizer()

        vOut = m * v

        self.assertEqual(vOut.Ex, v.Ex)
        self.assertEqual(vOut.Ey, 0)

    def testHorizontalPolarizerBlock(self):
        v = JonesVector(0, 1)
        m = HorizontalPolarizer()

        vOut = m * v

        self.assertEqual(vOut.Ex, v.Ex)
        self.assertEqual(vOut.Ey, 0)

    def testVerticalPolarizer(self):
        v = JonesVector(-1, 1)
        m = VerticalPolarizer()

        vOut = m * v

        self.assertEqual(vOut.Ex, 0)
        self.assertEqual(vOut.Ey, v.Ey)

    def testVerticalPolarizerBlock(self):
        v = JonesVector(-1, 0)
        m = VerticalPolarizer()

        vOut = m * v

        self.assertEqual(vOut.Ex, 0)
        self.assertEqual(vOut.Ey, v.Ey)

    def testHorizontalPolarizerRotatedBy90GivesVertical(self):
        h = HorizontalPolarizer().rotatedBy(pi / 2)
        v = VerticalPolarizer()

        self.assertAlmostEqual(h.A, v.A)
        self.assertAlmostEqual(h.B, v.B)
        self.assertAlmostEqual(h.C, v.C)
        self.assertAlmostEqual(h.D, v.D)

    def testPlus45Polarizer(self):
        v = JonesVector(1, 1)
        m = Plus45Polarizer()

        vOut = m * v

        self.assertEqual(vOut.Ex, v.Ex)
        self.assertEqual(vOut.Ey, v.Ey)

    def testPlus45PolarizerBlock(self):
        v = JonesVector(-1, 1)
        m = Plus45Polarizer()

        vOut = m * v

        self.assertEqual(vOut.Ex, 0)
        self.assertEqual(vOut.Ey, 0)

    def testMinus45Polarizer(self):
        v = JonesVector(-1, 1)
        m = Minus45Polarizer()

        vOut = m * v

        self.assertEqual(vOut.Ex, v.Ex)
        self.assertEqual(vOut.Ey, v.Ey)

    def testMinus45PolarizerBlock(self):
        v = JonesVector(1, 1)
        m = Minus45Polarizer()

        vOut = m * v

        self.assertEqual(vOut.Ex, 0)
        self.assertEqual(vOut.Ey, 0)

    def testRightCircularPolarizer(self):
        v = JonesVector(1, 1)
        m = RightCircularPolarizer()

        vOut = m * v

        self.assertTrue(vOut.isCircularlyPolarized)
        self.assertTrue(vOut.isRightCircularlyPolarized)
        self.assertFalse(vOut.isLeftCircularlyPolarized)

    def testLeftCircularPolarizer(self):
        v = JonesVector(1, 1)
        m = LeftCircularPolarizer()

        vOut = m * v

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
        ret = PhaseRetarder(delta=pi / 2)  # Ex is delayed by pi/2
        self.assertAlmostEqual(exp(1j * pi / 2), ret.A, 3)
        self.assertAlmostEqual(0, ret.B, 3)
        self.assertAlmostEqual(0, ret.C, 3)
        self.assertAlmostEqual(1, ret.D, 3)

        v = JonesVector(1, 1)
        vOut = ret * v

        self.assertTrue(vOut.isCircularlyPolarized)
        self.assertTrue(vOut.isRightCircularlyPolarized)
        self.assertFalse(vOut.isLeftCircularlyPolarized)

    def testQuarterWaveplateInit(self):
        qwp = QWP(theta=0)
        ret = PhaseRetarder(delta=-pi / 2)  # x is fast axis

        self.assertAlmostEqual(qwp.A, ret.A, 3)
        self.assertAlmostEqual(qwp.B, ret.B, 3)
        self.assertAlmostEqual(qwp.C, ret.C, 3)
        self.assertAlmostEqual(qwp.D, ret.D, 3)

    def testQuarterWaveplate(self):
        v = JonesVector(1, 1)
        m = QWP(theta=0)  # Ex is ahead = left circular

        vOut = m * v

        self.assertTrue(vOut.isCircularlyPolarized)
        self.assertTrue(vOut.isLeftCircularlyPolarized)
        self.assertFalse(vOut.isRightCircularlyPolarized)

        v = JonesVector(1, 1)
        m = QWP(theta=pi / 2)

        vOut = m * v

        self.assertTrue(vOut.isCircularlyPolarized)
        self.assertTrue(vOut.isRightCircularlyPolarized)
        self.assertFalse(vOut.isLeftCircularlyPolarized)

    def testHalfWaveplate(self):
        v = JonesVector(1, 1)
        m = HWP(theta=0)  # Ex is ahead by pi

        vOut = m * v

        self.assertAlmostEqual(vOut.Ex.real, -1, 5)
        self.assertAlmostEqual(vOut.Ey.real, 1, 5)

        v = JonesVector(1, 1)
        m = HWP(theta=pi / 2)  # Ey is ahead by pi

        vOut = m * v

        self.assertAlmostEqual(vOut.Ex.real, 1, 5)
        self.assertAlmostEqual(vOut.Ey.real, -1, 5)

        v = JonesVector(1, 0)
        m = HWP(theta=pi / 4)  # +45 is ahead by pi/4

        vOut = m * v

        self.assertAlmostEqual(vOut.Ex.real, 0, 5)
        self.assertAlmostEqual(vOut.Ey.real, -1, 5)

    def testRotation(self):
        v = Rotation(theta=pi / 2) * JonesVector(1, 0)
        self.assertAlmostEqual(abs(v.Ex), 0, 5)
        self.assertAlmostEqual(abs(v.Ey), 1, 5)

    @unittest.skip("Eigenvectors and eigenvalues need to be checked")
    def testRetarders(self):
        phi, e1, e2 = PhaseRetarder(delta=0.4).birefringence

        for theta in range(0, 90, 1):
            M = PhaseRetarder(delta=0.4).rotatedBy(theta * radPerDeg)
            delta, b1, b2 = M.birefringence
            if areAbsolutelyAlmostEqual(phi, delta):
                # FIXME: There is a discontinuity at 45°
                # It is not sufficient to flip b1 and b2
                self.assertAlmostEqual(phi, delta)
                self.assertAlmostEqual(b1.angleWith(e1, zHat), theta * radPerDeg)
            else:
                self.assertTrue(False, "Discontinuity at: {0} {1} {2} {3}".format(theta, delta, b1, b2))

    def testBirefringenceInWaveplates(self):
        self.assertTrue(QWP(theta=0).isBirefringent)
        self.assertTrue(QWP(theta=45 * radPerDeg).isBirefringent)
        self.assertTrue(QWP(theta=90 * radPerDeg).isBirefringent)
        self.assertTrue(QWP(theta=180 * radPerDeg).isBirefringent)

        self.assertTrue(HWP(theta=0).isBirefringent)
        self.assertTrue(HWP(theta=45 * radPerDeg).isBirefringent)
        self.assertTrue(HWP(theta=90 * radPerDeg).isBirefringent)
        self.assertTrue(HWP(theta=180 * radPerDeg).isBirefringent)

    def testNoBirefringenceInOtherMatrices(self):
        self.assertFalse(HorizontalPolarizer().isBirefringent)
        self.assertFalse(VerticalPolarizer().isBirefringent)
        self.assertFalse(Plus45Polarizer().isBirefringent)
        self.assertFalse(Minus45Polarizer().isBirefringent)
        self.assertFalse(RightCircularPolarizer().isBirefringent)
        self.assertFalse(LeftCircularPolarizer().isBirefringent)

    def testRotationNotBirefringent(self):
        m = HorizontalPolarizer().rotatedBy(theta=pi / 3)
        self.assertFalse(m.isBirefringent)

    @unittest.skip("ProcketCell need to be checked")
    def testPockelsCell(self):
        c = PockelsCell(halfwaveVoltage=300, length=10)
        c.voltage = 150
        self.assertAlmostEqual(c.retardance, pi / 2)
        c.voltage = -150
        self.assertAlmostEqual(c.retardance, -pi / 2)

    def testEigens(self):
        for theta in [1, 3, 5, 10, 80, 110, 130, 170, 189]:
            v1, v2, e1, e2 = Rotation(theta=theta * radPerDeg).eigens()
            self.assertTrue(isNotZero(v1.imag), v1)
            self.assertTrue(isNotZero(v2.imag), v2)
        for theta in [0, 180]:
            v1, v2, e1, e2 = Rotation(theta=theta * radPerDeg).eigens()
            self.assertTrue(isAlmostZero(v1.imag), v1)
            self.assertTrue(isAlmostZero(v2.imag), v2)

    def testArbitraryWavelengthDependentMatrix(self):
        mat = BirefringentMaterial(deltaIndex=0.1, fastAxisOrientation=0, physicalLength=1)
        with self.assertRaises(ValueError) as context:
            mat.m

        self.assertIsNotNone(mat.computeMatrix(k=2))
        anArray = mat.computeMatrix(k=2)
        self.assertAlmostEqual(angle(anArray[1, 1]) - angle(anArray[0, 0]), 0.2)

    def testArbitraryWavelengthDependentMatrixAbrOrientation(self):
        mat = BirefringentMaterial(deltaIndex=0.1, fastAxisOrientation=np.pi / 2, physicalLength=1)
        with self.assertRaises(ValueError) as context:
            mat.m
        self.assertEqual(mat.orientation, np.pi / 2)

        self.assertIsNotNone(mat.computeMatrix(k=2))
        anArray = mat.computeMatrix(k=2)
        self.assertAlmostEqual(angle(anArray[0, 0]) - angle(anArray[1, 1]), 0.2)

    def testBirefringentMaterialProductIsaMatrixProduct(self):
        mat1 = BirefringentMaterial(deltaIndex=0.1, fastAxisOrientation=np.pi / 2, physicalLength=1)
        mat2 = BirefringentMaterial(deltaIndex=0.1, fastAxisOrientation=np.pi / 2, physicalLength=1)
        final = mat1 * mat2
        self.assertTrue(isinstance(final, MatrixProduct))

        final2 = final * mat1
        self.assertTrue(isinstance(final2, MatrixProduct))

        final3 = mat1 * final
        self.assertTrue(isinstance(final3, MatrixProduct))

    def testBirefringentMaterialProductOfMatrixProductIsAMatrixProduct(self):
        mat1 = BirefringentMaterial(deltaIndex=0.1, fastAxisOrientation=np.pi / 2, physicalLength=1)
        mat2 = BirefringentMaterial(deltaIndex=0.1, fastAxisOrientation=np.pi / 2, physicalLength=1)
        final = mat1 * mat2
        final2 = final * mat1
        self.assertTrue(isinstance(final2, MatrixProduct))

    def testMatrixProductProductWithAMatrixProductIsAMatrixProduct(self):
        mat1 = HorizontalPolarizer()
        mat2 = BirefringentMaterial(deltaIndex=0.1, fastAxisOrientation=np.pi / 2, physicalLength=1)
        mat3 = BirefringentMaterial(deltaIndex=0.1, fastAxisOrientation=np.pi / 2, physicalLength=1)

        product12 = mat1 * mat2
        product23 = mat2 * mat3

        final = mat1 * mat2 * mat3
        final2 = product12 * mat3
        final3 = mat1 * product23

        self.assertEqual(final.matrices, final2.matrices)
        self.assertEqual(final.matrices, final3.matrices)
        self.assertEqual(final.matrices[0], mat3)
        self.assertEqual(final.matrices[1], mat2)
        self.assertEqual(final.matrices[2], mat1)

    def testBirefringentMaterialProductVector(self):
        mat = BirefringentMaterial(deltaIndex=0.1, fastAxisOrientation=np.pi / 2, physicalLength=1)
        v = JonesVector(1, 1, k=6.28)
        vOut = mat * v
        self.assertIsNotNone(vOut)
        self.assertEqual(v.k, vOut.k)

    def testBirefringentMaterialMatrixProductVector(self):
        mat1 = HorizontalPolarizer()
        mat2 = BirefringentMaterial(deltaIndex=0.1, fastAxisOrientation=np.pi / 2, physicalLength=1)
        mat3 = BirefringentMaterial(deltaIndex=0.2, fastAxisOrientation=np.pi / 2, physicalLength=1)
        v = JonesVector(1, 0, k=2)
        vOut = mat3 * v
        self.assertTrue(angle(vOut.Ex), 2 * 0.2)
        self.assertEqual(v.k, vOut.k)
        vOut = mat2 * mat3 * v
        self.assertTrue(angle(vOut.Ex), 2 * 0.3)
        self.assertEqual(v.k, vOut.k)
        product = mat2 * mat3
        vOut = product * v
        self.assertTrue(angle(vOut.Ex), 2 * 0.3)
        self.assertEqual(v.k, vOut.k)

    def testBackwardMatrix(self):
        mat1 = HorizontalPolarizer()
        self.assertIsNotNone(mat1.backward())


if __name__ == '__main__':
    unittest.main()
