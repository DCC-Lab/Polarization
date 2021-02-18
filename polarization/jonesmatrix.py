from numpy import complex, cos, sin, exp, array, pi, angle, matmul
from numpy.linalg import eig, det
from .utils import *
from .jonesvector import JonesVector
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class JonesMatrix:
    """ A Jones matrix represents any element that can transform polarization. 
    It is a complex 2 x 2 matrix that can transform a Jones vector
    (representing Ex and Ey).
    """
    def __init__(self, A: complex = 1, B: complex = 0, C: complex = 0, D: complex = 1, m=None, physicalLength: float = 0):
        """ We may initialize the matrix with all four elements or with a numpy
        array. It is mandatory to provide a length for the element, and it is
        assumed that the basis for the matrix is x̂ and ŷ.

        The basis is critical if we want to make sure we don't get confused
        It is very often implicitly x and y, but we will track it explicitly
        to avoid problems
        """
        if m is not None:
            self.mOriginal = array(m)
        else:
            self.mOriginal = array([[A,B],[C,D]])

        self.orientation = 0
        self.L = physicalLength

    @property
    def m(self):
        theta = self.orientation
        rotMatrix = array([[cos(theta),sin(theta)],[-sin(theta), cos(theta)]],dtype='complex')
        invRotMatrix = array([[cos(theta),-sin(theta)],[sin(theta), cos(theta)]],dtype='complex')

        return matmul(invRotMatrix, matmul(self.mOriginal, rotMatrix))

    @property
    def b1(self):
        """ The basis vector for x. For now this is not modifiable. """
        return array([1,0]) # x̂
    
    @property
    def b2(self):
        """ The basis vector for y. For now this is not modifiable. """
        return array([0,1]) # ŷ

    def setValue(self, name, value):
        try:
            setattr(self, name, value)
        except:
            print("Some properties are not mutable")

    def value(self, name):
        return getattr(self, name)

    @property
    def A(self):
        return self.m[0,0]
    @property
    def B(self):
        return self.m[0,1]
    @property
    def C(self):
        return self.m[1,0]
    @property
    def D(self):
        return self.m[1,1]

    @property
    def determinant(self):
        return det(self.m)
    
    @property
    def isBirefringent(self) -> bool:
        """ Returns True if it is birefringent.  See birefringence."""

        phi, e1, e2 = self.birefringence
        if isNotZero(phi, epsilon=1e-7):
            return True
        return False

    @property
    def birefringence(self) :
        """ Returns the birefringence parameters of this element. To determine
        these parameters, one must have a diagonalized version of this matrix
        and then determine if the diagonal element have a phase difference.

        To do that, we simply use a theorem of linear algebra: if a matrix is
        diagonalizable, then its eigenvectors are the appropriate basis and the
        associated eigenvalues are on the diagonal.

        TODO: Need to prove that the basis vectors will always be real.
        TODO: check e1 and e2 and get them to match x and y after rotation

        """
        w, v = eig(self.m)
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
    def retardance(self) -> float:
        phi, e1, e2 = self.birefringence
        return phi
    
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
            including the 2 x 2 matrix and other properties of an element.

        Returns
        -------
        A matrix with:

        a : float
            Value of the index (1,1) in the 2 x 2 matrix of the combination of the two elements.
        b : float
            Value of the index (2,1) in the 2 x 2 matrix of the combination of the two elements.
        c : float
            Value of the index (1,2) in the 2 x 2 matrix of the combination of the two elements.
        d : float
            Value of the index (2,2) in the 2 x 2 matrix of the combination of the two elements.

        """

        product = JonesMatrix(m=matmul(self.m, rightSideMatrix.m), physicalLength=self.L + rightSideMatrix.L)
        return product

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

    def rotatedBy(self, theta):
        """ We return a rotated copy of the optical element of the matrix by theta. 
        For instance, a theta rotation of a horizontal polarizer will be a polarizer
        aligned at theta. """

        return Rotation(theta = -theta)*self*Rotation(theta = theta)

    def rotateBasisBy(self, theta):
        """ We rotate the basis of the matrix by theta. This is different 
        from `rotateEffectBy` because it does not change the effect, only the
        way it is represented. For instance, a 45 rotation of a horizontal
        polarizer will be a polarizer *still* aligned horizontally, but
        a basis x,y that has been rotated to +45 and +135."""
        raise NotImplemented()

    def show(self, input, xObj, xProperty, xRange, yObj, yProperty):
        if xObj is None:
            xObj = self

        x = []
        y = []

        for value in xRange:
            if xObj is None:
                self.setValue(xProperty, value)
            else:
                xObj.setValue(xProperty, value)

            vOut = self*input

            x.append( xObj.value(xProperty))
            if yObj is None:
                y.append( vOut.value(yProperty))
            else:
                y.append( yObj.value(yProperty))
        
        plt.title("{0} versus {1}".format(yProperty, yProperty))
        plt.xlabel("{0}".format(xProperty))
        plt.ylabel("{0}".format(yProperty))
        plt.plot(x,y,'ko')
        plt.show()


    def showOrientationPlot(self, input:JonesVector):
        x = []
        y = []
        for theta in range(0,190,10):
            theMatrix = JonesMatrix(m=self.m)
            theMatrix.orientation = theta*radPerDeg

            vOut = theMatrix*input
            x.append(theta)
            y.append(vOut.intensity)

        plt.title("Intensity versus orientation of element")
        plt.xlabel(r"Rotation $\theta$ of element [°]")
        plt.ylabel(r"Intensity [arb. unit]")
        plt.plot(x,y,'ko')
        plt.show()

    def showPolarizationPlot(self):
        x = []
        y = []
        for theta in range(0,190,10):
            vIn = JonesVector(Ex=cos(theta*radPerDeg), Ey=sin(theta*radPerDeg))
            vOut = self*vIn
            x.append(theta)
            y.append(vOut.intensity)

        plt.title("Intensity versus input polarization orientation (0° is horizontal)")
        plt.xlabel(r"Input polarization orientation [°]")
        plt.xlim(0,180)
        plt.ylabel(r"Intensity [arb. unit]")
        plt.plot(x,y,'ko')
        plt.show()


class HorizontalPolarizer(JonesMatrix):
    def __init__(self):
        JonesMatrix.__init__(self, A=1,B=0,C=0,D=0,physicalLength=0)        

class LinearPolarizer(JonesMatrix):
    def __init__(self, theta):
        c = cos(theta)
        s = sin(theta)
        JonesMatrix.__init__(self, A=c*c,B=c*s,C=c*s,D=s*s,physicalLength=0)

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
    def __init__(self, delta=None, phiX=None, phiY=None, theta=0, physicalLength=0):
        if delta is not None:
            JonesMatrix.__init__(self, A=exp(1j * delta), B=0, C=0, D=1, physicalLength=0)
        else:
            JonesMatrix.__init__(self, A=exp(1j * phiX), B=0, C=0, D=exp(1j * phiY), physicalLength=0)

    
class QWP(PhaseRetarder):
    def __init__(self, theta):
        # theta is fast axis with respect to x-axis
        PhaseRetarder.__init__(self, delta=-pi / 2) # Ex is advanced by pi/2, x is fast
        self.orientation = theta

class HWP(PhaseRetarder):
    def __init__(self, theta):
        PhaseRetarder.__init__(self, delta=-pi) # Ex is advanced by pi, x is fast
        self.orientation = theta

class PockelsCell(JonesMatrix):
    def __init__(self, halfwaveVoltage, length):
        self.halfwaveVoltage = halfwaveVoltage
        self.voltage = 0
        JonesMatrix.__init__(self, A=1, B=0, C=0, D=1, physicalLength=length)

    @property
    def m(self):
        cell = PhaseRetarder(delta=self.voltage/self.halfwaveVoltage*pi)
        cell.orientation = self.orientation
        return cell.m

    def showVoltagePlot(self):
        fig, axs = plt.subplots(1, sharex=True)
        fig.suptitle("Pockels cell at {0:.1f}° with horizontal incident polarization".format(self.orientation*degPerRad))

        voltages = list(range(0,self.halfwaveVoltage+10,10))
        yParallel = []
        yCrossed = []
        for voltage in voltages:
            self.voltage = voltage

            vIn = JonesVector.horizontal()
            vOut = HorizontalPolarizer()*self*vIn
            yParallel.append(vOut.intensity)

            vOut = VerticalPolarizer()*self*vIn
            yCrossed.append(vOut.intensity)


        # axs[0].xlim(0, self.halfwaveVoltage)
        # fig.ylabel(r"Intensity [arb. unit]")
        axs.plot(voltages,yParallel,'k|',label="Between parallel polarizers")
        axs.plot(voltages,yCrossed,'k+',label="Between crossed polarizers")
        axs.legend()
        plt.show()


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
