from numpy import complex, cos, sin, exp, array, pi, angle, matmul
from numpy.linalg import eig, det
from .utils import *
from .jonesvector import JonesVector
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .vector import *

class JonesMatrix:
    """ A Jones matrix represents any element that can transform polarization. 
    It is a complex 2 x 2 matrix that can transform a Jones vector
    (representing Ex and Ey).
    """
    def __init__(self, A: complex = None,
                       B: complex = None,
                       C: complex = None,
                       D: complex = None, 
                       m=None, 
                       physicalLength: float = 0):
        """ We may initialize the matrix with all four elements or with a numpy
        array. It is mandatory to provide a length for the element, and it is
        assumed that the basis for the matrix is x̂ and ŷ.

        The basis (b1, b2) is critical if we want to make sure we don't get confused
        It is very often implicitly x and y, but we will track it explicitly
        to avoid problems. For now, it is x and y.
        """
        if m is not None:
            self.mOriginal = array(m)
        elif A is not None and D is not None and C is not None and D is not None:
            self.mOriginal = array([[A,B],[C,D]])
        else:
            # Obviously, the subclass will compute when the time comes
            # See Birefringence() for an example
            self.mOriginal = None

        self.orientation = 0
        self.L = physicalLength

    def mNumeric(self, k=None):
        if self.mOriginal is None:
            # This is the signal that the matrix depends on k.
            # subclasses will calculate m as a function of k when needed
            # by overriding this function.
            raise ValueError('This matrix {0} appears to be wavelength-dependent. \
You cannot obtain the values without providing a wavevector k or the matrix itself.'.format(type(self)))

        theta = self.orientation
        rotMatrix = array([[cos(theta),sin(theta)],[-sin(theta), cos(theta)]],dtype='complex')
        invRotMatrix = array([[cos(theta),-sin(theta)],[sin(theta), cos(theta)]],dtype='complex')

        return matmul(invRotMatrix, matmul(self.mOriginal, rotMatrix))

    @property
    def m(self):
        # Most Jones matrices do not depend on k, so we can return the matrix.
        # Some subclasses *may* depend on k.
        return self.mNumeric(k=None)
    
    @property
    def b1(self):
        """ The basis vector for x. For now this is not modifiable. 
        b1 x b2 = direction of propagation """
        return Vector(1,0,0) # x̂
    
    @property
    def b2(self):
        """ The basis vector for y. For now this is not modifiable. 
        b1 x b2 = direction of propagation """
        return Vector(0,1,0) # ŷ

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

        phi1, phi2, e1, e2 = self.birefringence
        if isNotZero(phi1-phi2, epsilon=1e-7):
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

        We always want b1 x b2 = direction of propagation.  Therefore, before
        leaving, we quickly check that it is the case, and swap them
        (and change the sign of phi) if not the case.

        TODO: Need to prove that the basis vectors will always be real
        if matrices are always symmetric/hermitian. Not sure for the general
        case.

        """

        (w1, w2, e1, e2) = self.eigens()

        phi1 = angle(w1)
        phi2 = angle(w2)

        v1 = None
        v2 = None
        if e1 is not None:
            v1 = Vector(e1[0], e1[1], 0)

        if e2 is not None:
            v2 = Vector(e2[0], e2[1], 0)
        elif e1 is not None:
            v2 = Vector(v1.y, -v1.x, 0)

        return phi1, phi2, v1, v2

    def eigens(self):
        """ This returns the eigenvalues and eigenvectors.  It attempts to keep
        the eigenvectors real by transferring a j factor to the eigenvalue if
        the eigenvector is imaginary """

        w, v = eig(self.m)

        e1 = realIfPossible(v[0])
        e2 = realIfPossible(v[1])

        return (w[0], w[1], e1, e2)

    @property
    def isOpticallyActive(self) -> bool:
        return False

    @property
    def diattenuation(self) -> complex:
        (w1, w2, e1, e2) = self.eigens()

        t1 = abs(w1)
        t2 = abs(w2)

        return t1, t2, e1, e2

    @property
    def retardance(self) -> float:
        phi1, phi2, e1, e2 = self.birefringence
        return phi1-phi2
    
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
        elif isinstance(rightSide, number_types):
            return self.mul_number(rightSide)
        else:
            raise TypeError(
                "Unrecognized right side element in multiply: '{0}'\
                 cannot be multiplied by a JonesMatrix".format(rightSide))

    def __pow__(self, power, modulo=None):
        """ Matrix power multiplication. """
        outputMatrix = self
        for _ in range(power-1):
            outputMatrix = self.mul_matrix(outputMatrix)
        return outputMatrix

    def __rmul__(self, leftSide):
        """ Multiplies a Jones matrix object by a number. """
        if isinstance(leftSide, number_types):
            return self.mul_number(leftSide)
        else:
            raise TypeError(
                "Unrecognized left side element in multiply: '{0}'\
                 cannot be multiplied by a JonesMatrix".format(leftSide))

    def __truediv__(self, other):
        """ Divides a Jones matrix by a number. """
        if isinstance(other, number_types):
            return self.mul_number(1/other)
        else:
            raise TypeError(
                "Unrecognized number in true divide: '{0}'\
                 cannot be multiplied by a JonesMatrix".format(other))

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
        self.k = rightSideVector.k #FIXME: This is hacky. We should use JonesMatrixGroup
        outputVector.Ex = self.A * rightSideVector.Ex + self.B * rightSideVector.Ey
        outputVector.Ey = self.C * rightSideVector.Ex + self.D * rightSideVector.Ey
        outputVector.z = self.L + rightSideVector.z
        self.k = None
        return outputVector

    def mul_number(self, n):
        """ Multiply a Jones matrix by a number."""
        return JonesMatrix(self.A*n, self.B*n, self.C*n, self.D*n, physicalLength=self.L)

    def __add__(self, other):
        """ Adds two Jones matrices. """
        if isinstance(other, JonesMatrix):
            return self.add_matrix(other)
        elif isinstance(other, number_types):
            return self.add_number(other)
        else:
            raise TypeError(
                "Unrecognized element in addition: '{0}'\
                 cannot be added to a JonesMatrix".format(other))

    def __sub__(self, other):
        """ Subtracts two Jones matrices. """
        if isinstance(other, JonesMatrix):
            return self.add_matrix(other * -1)
        else:
            raise TypeError(
                "Unrecognized element in addition: '{0}'\
                 cannot be added to a JonesMatrix".format(other))

    def add_matrix(self, rightSideMatrix: 'JonesMatrix'):
        a = self.A + rightSideMatrix.A
        b = self.B + rightSideMatrix.B
        c = self.C + rightSideMatrix.C
        d = self.D + rightSideMatrix.D

        return JonesMatrix(a, b, c, d, physicalLength=self.L)

    def add_number(self, n):
        return JonesMatrix(self.A + n, self.B + n, self.C + n, self.D + n, physicalLength=self.L)

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

    def setValue(self, name, value):
        try:
            setattr(self, name, value)
        except:
            raise Exception("Some properties are not mutable")

    def value(self, name):
        return getattr(self, name)

    def show(self, input, xObj, xProperty, xRange, yObj, yProperty): # pragma: no cover
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

    def showOrientationPlot(self, input:JonesVector): # pragma: no cover
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

    def showPolarizationPlot(self): # pragma: no cover
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

    def __str__(self):
        return "[[{}, {}],\n[{}, {}]]\n".format(self.A, self.B, self.C, self.D)


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
    def __init__(self, delta=None, phiX=None, phiY=None, physicalLength=0):
        if delta is not None:
            JonesMatrix.__init__(self, A=exp(1j * delta), B=0, C=0, D=1, physicalLength=physicalLength)
        else:
            JonesMatrix.__init__(self, A=exp(1j * phiX), B=0, C=0, D=exp(1j * phiY), physicalLength=physicalLength)

class BirefringentMaterial(JonesMatrix):
    def __init__(self, deltaIndex:float, orientation, physicalLength=0):
        JonesMatrix.__init__(self, A=None, B=None, C=None, D=None, physicalLength=physicalLength)
        self.deltaIndex = deltaIndex
        self.orientation = orientation

    def mNumeric(self, k=None):
        if k is not None:
            phi = k * self.deltaIndex * self.physicalLength
            explicit = JonesMatrix(A=exp(1j * phi), B=0, C=0, D=1, physicalLength= self.physicalLength)
            return explicit.mNumeric()
        else:
            raise ValueError("You must provide k for this matrix")


class Diattenuator(JonesMatrix):
    def __init__(self, Tx, Ty, physicalLength=0):
        JonesMatrix.__init__(self, A=Tx, B=0, C=0, D=Ty, physicalLength=0)
    
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

    def showVoltagePlot(self): # pragma: no cover
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
