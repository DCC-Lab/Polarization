import numpy as np
from .jonesvector import JonesVector

class JonesMatrix:
    def __init__(self, A: np.complex = 1, B: np.complex = 0, C: np.complex = 0, D: np.complex = 1, physicalLength: np.float = 0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.L = physicalLength

    @property
    def determinant(self):
        return self.A*self.D-self.B-self.C
    
    @property
    def isBirefringent(self) -> np.bool:
        return False

    @property
    def isOpticallyActive(self) -> np.bool:
        return False

    @property
    def diattenuation(self) -> np.complex:
        return np.complex(0, 0)

    @property
    def retardance(self) -> np.complex:
        return np.complex(0, 0)
    
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
        JonesMatrix.__init__(self, A=np.cos(theta), B=np.sin(theta), C=-np.sin(theta), D=np.cos(theta), physicalLength=0)

class PhaseRetarder(JonesMatrix):
    def __init__(self, delta=None, phiX=None, phiY=None, physicalLength=0):
        if delta is not None:
            JonesMatrix.__init__(self, A=np.exp(1j * delta), B=0, C=0, D=1, physicalLength=0)
        else:
            JonesMatrix.__init__(self, A=np.exp(1j * phiX), B=0, C=0, D=np.exp(1j * phiY), physicalLength=0)

class QWP(JonesMatrix):
    def __init__(self, theta):
        # theta is fast axis with respect to x-axis
        baseChange = Rotation(theta)
        retardance = PhaseRetarder(delta=-np.pi / 2) # Ex is advanced by pi/2, x is fast
        invBaseChange = Rotation(-theta)

        qwp = invBaseChange*retardance*baseChange
        JonesMatrix.__init__(self, A=qwp.A, B=qwp.B, C=qwp.C, D=qwp.D, physicalLength=0)        

class HWP(JonesMatrix):
    def __init__(self, theta):
        baseChange = Rotation(theta)
        retardance = PhaseRetarder(delta=-np.pi)  # Ex is advanced by pi, x is fast
        invBaseChange = Rotation(-theta)

        hwp = invBaseChange*retardance * baseChange
        JonesMatrix.__init__(self, A=hwp.A, B=hwp.B, C=hwp.C, D=hwp.D,physicalLength=0)


class Retarder(JonesMatrix):  # fixme: don't know how to call a JonesMatrixFromRetardanceAndDiattenuation
    def __init__(self, retardance, diattenuation=None):
        if diattenuation is None:
            diattenuation = np.zeros(retardance.shape)
        dim = retardance.shape
        f = (diattenuation - 1j * retardance) / 2
        c = np.sqrt(np.sum(f ** 2, axis=0)).reshape(1, -1)
        sinch = np.sinh(c) / c
        sinch[c == 0] = 1
        jonesMat = np.array([[1], [0], [0], [1]]) * (np.cosh(c)) + sinch * (
                np.array([[1], [0], [0], [-1]]) * f[0, :].reshape(1, -1) +
                np.array([[0], [1], [1], [0]]) * f[1, :].reshape(1, -1) +
                np.array([[0], [1j], [-1j], [0]]) * f[2, :].reshape(1, -1))
        if np.size(retardance) == 3:
            jonesMat = jonesMat.reshape((2, 2))
        else:
            jonesMat = np.squeeze(jonesMat.reshape(2, 2, dim[1], -1))
        # return jonesMat
