from numpy import complex, cos, sin, exp, array, pi, angle
from numpy.linalg import eig
from .utils import *
from .jonesvector import JonesVector

class JonesMatrix:
    def __init__(self, A: complex = 1, B: complex = 0, C: complex = 0, D: complex = 1, physicalLength: float = 0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.L = physicalLength

        # The basis is critical if we want to make sure we don't get confused
        # It is very often implicitly x and y, but we will track it explicitly
        # to avoid problems

        self.b1 = (1,0)
        self.b2 = (0,1)

    @property
    def asArray(self):
        return array([[self.A, self.B],[self.C, self.D]])

    def setFromArray(self, anArray):
        self.A, self.B, self.C, self.D = anArray

    @property
    def determinant(self):
        return self.A*self.D-self.B-self.C
    
    @property
    def isBirefringent(self) -> bool:
        phi, e1, e2 = self.birefringence
        if isNotZero(phi, epsilon=1e-7):
            return True
        return False

    @property
    def birefringence(self) :
        w, v = eig(self.asArray)
        phi = angle(w[0]) - angle(w[1])
        e1 = [1,0]
        if isEssentiallyReal(v[0][0]):
            e1[0] = v[0][0].real
        if isEssentiallyReal(v[0][1]):
            e1[1] = v[0][1].real

        if isAlmostZero(e1[0]):
            e1[0] = 0
        if isAlmostZero(e1[1]):
            e1[1] = 0


        e2 = [0,1]
        if isEssentiallyReal(v[1][0]):
            e2[0] = v[1][0].real
        if isEssentiallyReal(v[1][1]):
            e2[1] = v[1][1].real
        if isAlmostZero(e2[0]):
            e2[0] = 0
        if isAlmostZero(e2[1]):
            e2[1] = 0

        return phi, e1, e2

    @property
    def isOpticallyActive(self) -> bool:
        return False

    @property
    def diattenuation(self) -> complex:
        return complex(0, 0)

    @property
    def retardance(self) -> complex:
        return complex(0, 0)
    
    def __mul__(self, rightSide):
        """Operator overloading allowing easy-to-read matrix multiplication
        with other `JonesMatrix`, with a `JonesVector`.

        For instance, with M1 = JonesMatrix() and M2 = JonesMatrix(), one can write
        M3 = M1*M2. With r = JonesVector(), one can apply the M1 transform to a vector
        with rOut = M1*r

        """
        if isinstance(rightSide, JonesMatrix):
            return self.mul_matrix(rightSide)
        elif isinstance(rightSide, JonesVector):
            return self.mul_vector(rightSide)
        else:
            raise TypeError(
                "Unrecognized right side element in multiply: '{0}'\
                 cannot be multiplied by a JonesMatrix".format(rightSide))

    def mul_matrix(self, rightSideMatrix: 'JonesMatrix'):
        r""" This function is used to combine two elements into a single matrix.


        Parameters
        ----------
        rightSideMatrix : object from Matrix class
            including the ABCD matrix and other properties of an element.

        Returns
        -------
        A matrix with:

        a : float
            Value of the index (1,1) in the ABCD matrix of the combination of the two elements.
        b : float
            Value of the index (2,1) in the ABCD matrix of the combination of the two elements.
        c : float
            Value of the index (1,2) in the ABCD matrix of the combination of the two elements.
        d : float
            Value of the index (2,2) in the ABCD matrix of the combination of the two elements.

        """

        a = self.A * rightSideMatrix.A + self.B * rightSideMatrix.C
        b = self.A * rightSideMatrix.B + self.B * rightSideMatrix.D
        c = self.C * rightSideMatrix.A + self.D * rightSideMatrix.C
        d = self.C * rightSideMatrix.B + self.D * rightSideMatrix.D
        L = self.L + rightSideMatrix.L

        return JonesMatrix(a, b, c, d, physicalLength=L)

    def mul_vector(self, rightSideVector):
        r"""This function does the multiplication of a vector by a matrix.
        The output shows the propagated ray through the system.
        New position of ray is updated by the physical length of the matrix.

        Parameters
        ----------
        rightSideVector : object from JonesVector class

        Returns
        -------
        outputVector : an object from JonesVector class
            New position of the input vector after passing through the element.

        """

        outputVector = JonesVector()
        outputVector.Ex = self.A * rightSideVector.Ex + self.B * rightSideVector.Ey
        outputVector.Ey = self.C * rightSideVector.Ex + self.D * rightSideVector.Ey
        outputVector.z = self.L + rightSideVector.z

        return outputVector

    def rotateEffectBy(self, theta):
        return self*Rotation(theta = theta)

    def rotateBasisBy(self, theta):
        raise NotImplemented()

class HorizontalPolarizer(JonesMatrix):
    def __init__(self):
        JonesMatrix.__init__(self, A=1,B=0,C=0,D=0,physicalLength=0)        

class VerticalPolarizer(JonesMatrix):
    def __init__(self):
        JonesMatrix.__init__(self, A=0,B=0,C=0,D=1,physicalLength=0)        

class Plus45Polarizer(JonesMatrix):
    def __init__(self):
        JonesMatrix.__init__(self, A=0.5,B=0.5,C=0.5,D=0.5,physicalLength=0)        

class Minus45Polarizer(JonesMatrix):
    def __init__(self):
        JonesMatrix.__init__(self, A=0.5,B=-0.5,C=-0.5,D=0.5,physicalLength=0)        

class RightCircularPolarizer(JonesMatrix):
    def __init__(self):
        JonesMatrix.__init__(self, A=0.5,B=0.5j,C=-0.5j,D=0.5,physicalLength=0)        

class LeftCircularPolarizer(JonesMatrix):
    def __init__(self):
        JonesMatrix.__init__(self, A=0.5,B=-0.5j,C=0.5j,D=0.5,physicalLength=0)        

class Rotation(JonesMatrix):
    def __init__(self, theta):
        JonesMatrix.__init__(self, A=cos(theta), B=sin(theta), C=-sin(theta), D=cos(theta), physicalLength=0)

class PhaseRetarder(JonesMatrix):
    def __init__(self, delta=None, phiX=None, phiY=None, physicalLength=0):
        if delta is not None:
            JonesMatrix.__init__(self, A=exp(1j * delta), B=0, C=0, D=1, physicalLength=0)
        else:
            JonesMatrix.__init__(self, A=exp(1j * phiX), B=0, C=0, D=exp(1j * phiY), physicalLength=0)

class QWP(JonesMatrix):
    def __init__(self, theta):
        # theta is fast axis with respect to x-axis
        baseChange = Rotation(theta)
        retardance = PhaseRetarder(delta=-pi / 2) # Ex is advanced by pi/2, x is fast
        invBaseChange = Rotation(-theta)

        qwp = invBaseChange*retardance*baseChange
        JonesMatrix.__init__(self, A=qwp.A, B=qwp.B, C=qwp.C, D=qwp.D, physicalLength=0)        

class HWP(JonesMatrix):
    def __init__(self, theta):
        baseChange = Rotation(theta)
        retardance = PhaseRetarder(delta=-pi)  # Ex is advanced by pi, x is fast
        invBaseChange = Rotation(-theta)

        hwp = invBaseChange*retardance * baseChange
        JonesMatrix.__init__(self, A=hwp.A, B=hwp.B, C=hwp.C, D=hwp.D,physicalLength=0)


# class Retarder(JonesMatrix):  # fixme: don't know how to call a JonesMatrixFromRetardanceAndDiattenuation
#     def __init__(self, retardance, diattenuation=None):
#         if diattenuation is None:
#             diattenuation = np.zeros(retardance.shape)
#         dim = retardance.shape
#         f = (diattenuation - 1j * retardance) / 2
#         c = np.sqrt(np.sum(f ** 2, axis=0)).reshape(1, -1)
#         sinch = sinh(c) / c
#         sinch[c == 0] = 1
#         jonesMat = array([[1], [0], [0], [1]]) * (cosh(c)) + sinch * (
#                 array([[1], [0], [0], [-1]]) * f[0, :].reshape(1, -1) +
#                 array([[0], [1], [1], [0]]) * f[1, :].reshape(1, -1) +
#                 array([[0], [1j], [-1j], [0]]) * f[2, :].reshape(1, -1))
#         if np.size(retardance) == 3:
#             jonesMat = jonesMat.reshape((2, 2))
#         else:
#             jonesMat = np.squeeze(jonesMat.reshape(2, 2, dim[1], -1))
#         # return jonesMat
