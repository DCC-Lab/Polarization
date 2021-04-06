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
                       physicalLength: float = 0,
                       orientation: float = 0):
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
            # See BirefringenceMaterial() for an example
            self.mOriginal = None

        self.orientation = orientation
        self.L = physicalLength

        """ The basis vector for x and y. For now this is not really
        modifiable.  b1 x b2 = b3, direction of propagation """
        self.b1 = Vector(1,0,0) # x̂
        self.b2 = Vector(0,1,0) # ŷ
        self.b3 = Vector(0,0,1) # ẑ

    def computeMatrix(self, k=None, l=None, backward=bool):
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
    
    def backward(self):
        """ Return a matrix for a JonesVector propagating in the opposite
        direction. Flip the direction of propagation that is assumed when  getting the
        matrix.  By default, upon creation, we assume  the beam propagates along +z. 
        When a beam is reflected, its direction will change from +z to -z, but the
        matrices used for the calculation must also change. For now, we will simply 
        assume the matrix is the transpose, which is not always right (only true for
        reciprocal matrices). Tests set up for this may fail.


        FIXME: Right now, this will fail for matrices such as BirefringentMaterial
        or Faraday rotator.
        FIXME: I am not sure about the orientation.
        """

        backward = JonesMatrix(m=self.m.T,
                               physicalLength=self.L,
                               orientation=self.orientation)
        backward.b3 = -backward.b3
        backward.b2 = -backward.b2
        backward.orientation = -backward.orientation
        return backward

    @property
    def m(self):
        return self.computeMatrix()
    
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
        elif isinstance(rightSide, MatrixProduct):
            product = MatrixProduct(matrices=rightSide.matrices)
            product.append(self)
            return product
        elif isinstance(rightSide, JonesVector):
            return self.mul_vector(rightSide)
        elif isinstance(rightSide, number_types):
            return self.mul_number(rightSide)
        else:
            raise TypeError(
                "Unrecognized right side element in multiply: '{0}'\
                 cannot be multiplied by a JonesMatrix".format(rightSide))

    def __rmul__(self, leftSide):
        """ Multiplies a Jones matrix object by a number. """
        if isinstance(leftSide, number_types):
            return self.mul_number(leftSide)
        else:
            raise TypeError(
                "Unrecognized left side element in multiply: '{0}'\
                 cannot be multiplied by a JonesMatrix".format(leftSide))
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

        try:
            theMatrix = self.computeMatrix()
            product = JonesMatrix(m=matmul(theMatrix, rightSideMatrix.m), physicalLength=self.L + rightSideMatrix.L)
            return product
        except ValueError as err:
            # There is no possible numerical value at this point. Let's return an
            # object the holds the elements together to be multiplied later
            # when we know the JonesVector
            return MatrixProduct( [rightSideMatrix, self])


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
        # We obtain the matrix specific to this JonesVector
        m = self.computeMatrix(k = rightSideVector.k)
        
        # Is the matrix in the appropriate direction?
        # For now, print a warning.
        if rightSideVector.b3 != self.b3:
            print("Warning: the matrix is set explicitly set up for propagation along the opposite direction of the JonesVector")

        direction = zHat.dot(rightSideVector.b3) # +1 or -1

        outputVector.Ex = m[0,0] * rightSideVector.Ex + m[0,1] * rightSideVector.Ey
        outputVector.Ey = m[1,0] * rightSideVector.Ex + m[1,1] * rightSideVector.Ey
        outputVector.z = rightSideVector.z + direction*self.L
        outputVector.k = rightSideVector.k
        return outputVector

    def mul_number(self, n):
        """ Multiply a Jones matrix by a number."""
        return JonesMatrix(self.A*n, self.B*n, self.C*n, self.D*n, physicalLength=self.L)

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
        except Exception as err:
            raise Exception("Some properties are not mutable: {0}".format(err))

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
    def __init__(self, deltaIndex:float, fastAxisOrientation, physicalLength=0):
        """ The fast axis is the X axis when fastAxisOrientation = 0"""
        JonesMatrix.__init__(self, A=None, B=None, C=None, D=None, physicalLength=physicalLength, orientation=fastAxisOrientation)
        self.deltaIndex = deltaIndex
        self.isBackward = False

    def computeMatrix(self, k=None):
        if k is not None:
            phi = k * self.deltaIndex * self.L
            explicit = JonesMatrix(A=1, B=0, C=0, D=exp(1j * phi), physicalLength=self.L)
            explicit.orientation = self.orientation
            if self.isBackward:
                explicit = JonesMatrix(m=explicit.m.T, physicalLength=self.L, orientation=-self.orientation)
                explicit.b3 = -explicit.b3
                explicit.b2 = -explicit.b2
            return explicit.computeMatrix()
        else:
            raise ValueError("You must provide k for this matrix")

    def backward(self):
        backward = BirefringentMaterial(deltaIndex=self.deltaIndex,
                                        fastAxisOrientation=self.orientation,
                                        physicalLength=self.L)
        backward.isBackward = True
        return backward

class Diattenuator(JonesMatrix):
    def __init__(self, Tx, Ty, physicalLength=0):
        JonesMatrix.__init__(self, A=Tx, B=0, C=0, D=Ty, physicalLength=physicalLength)
    
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


class MatrixProduct:
    def __init__(self, matrices=None):
        """ Sometimes we may have an expression that involves the
        multiplication of matrices but one of those matrices depends
        on k, and we don't know yet because we have not multiplied by
        a JonesVector (which has a k property). Hence, we keep the
        product inside a `MatrixProduct` so that when the time comes
        we can get the numerical value for the matrix and multiply
        them properly.

        Matrices that will multiply a JonesVector at some point
        The first matrix is the first that will multiply so it is the
        rightmost matrix.  The last matrix in the array is the leftmost
        matrix.
        """

        self.matrices = []
        if matrices is not None:
            for matrix in matrices:
                self.append(matrix)

    def append(self, matrix: 'JonesMatrix'):
        self.matrices.append(matrix)

    def __mul__(self, rightSide):
        """ We have a MatrixProduct being multiplied by something else.
        If this is another matrix, we simply prepend or append the object
        depending on the fact that it may be multiplied left/right. We still
        need to return a MatrixProduct, because we still don't know the k vector
        at this point.
        The key moment is when we multiply this MatrixProduct by a JonesVector:
        our mul_vector method will unwrap the whole product and return, finally, a numerical value
        since it now has access to k and can compute the numerical values for the matrices.
        """

        if isinstance(rightSide, MatrixProduct):
            product = MatrixProduct()
            product.matrices.extend(rightSide.matrices)
            product.matrices.extend(self.matrices)
            return product
        elif isinstance(rightSide, JonesMatrix):
            product = MatrixProduct()
            product.matrices.append(rightSide)
            product.matrices.extend(self.matrices)
            return product
        elif isinstance(rightSide, JonesVector):
            return self.mul_vector(rightSide)
        else:
            raise TypeError("Unknown type mul")

    def mul_vector(self, vector):
        """ At this point, we are multiplying the MatrixProduct by a
        JonesVector, therefore we *know* the wavevector k for the multiplication. By
        managing the product ourselves, we start the multiplication "from the right"
        and multiply the rightmost matrix by the JonesVector , and if that  matrix
        requires the vector k, it will request it in mul_vector in order to calculate
        the numerical value of the matrix. """
        outputVector = JonesVector(Ex=vector.Ex, Ey=vector.Ey, k=vector.k, z=vector.z)
        for m in self.matrices:
            outputVector = m*outputVector

        return outputVector

