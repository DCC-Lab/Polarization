from numpy import complex, exp, array, sqrt, cos, sin, arctan2, pi, conj, abs, angle
from .utils import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .vector import *

class JonesVector:    
    def __init__(self, Ex: complex = 0.0, Ey: complex = 0.0, wavelength:float = None, k:float = None, z= 0 ):
        """ The phase of the field is (k*z-omega*t+phi).
        A positive phase phi is a delayed field, and a negative
        phase is an advanced field. We always assume that the basis 
        is x,y when initializing. This can be changed later
        but we provide Ex and Ey as read-only properties.

        See https://en.wikipedia.org/wiki/Jones_calculus

        """
        self.E1 = complex(Ex)
        self.E2 = complex(Ey)
        self.z = z
        if k is not None and wavelength is not None:
            raise ValueError('Provide either one of wavelength or k, but not both')
        elif k is not None:
            self.k = k
        elif wavelength is not None:
            self.k = 2*np.pi/wavelength
        else:
            # If at any point in calculation k is needed, it will fail
            # This is the expected behaviour: k must be set explicitly
            # when required in calculations
            self.k = None 

        """ The basis vector for E1 and E2. We settled for b1, b2 and b3. For
        now this is not modifiable. """
        self.b1 = XYVector(1,0) # x̂ for E1
        self.b2 = XYVector(0,1) # ŷ for E2
        self.b3 = ZVector(1) # ẑ direction propagation b1 x b2 == b3

    @property
    def Ex(self):
        return self.E1 * self.b1.x + self.E2 * self.b2.x
    
    @Ex.setter
    def Ex(self, value):
        if self.b1 == xHat:
            self.E1 = value
        elif self.b2 == xHat:
            self.E2 = value
        else:
            raise RuntimeError("Unable to set Ey if one of the basis vectors is not x̂")

    @property
    def Ey(self):
        return self.E1 * self.b1.y + self.E2 * self.b2.y

    @Ey.setter
    def Ey(self, value):
        if self.b2 == yHat:
            self.E2 = value
        elif self.b1 == yHat:
            self.E1 = value
        else:
            raise RuntimeError("Unable to set Ey if one of the basis vectors is not ŷ")

    def setValue(self, name, value):
        try:
            setattr(self, name, value)
        except:
            print("Some properties are not mutable")

    def value(self, name):
        return getattr(self, name)

    def normalize(self):
        """ Normalize the field amplitudes to obtain an intensity of 1 """

        fieldAmplitude = sqrt(self.intensity)
        if fieldAmplitude != 0:
            self.E1 /= fieldAmplitude
            self.E2 /= fieldAmplitude
        return self

    @property
    def orientation(self):
        """ Orientation of the polarization ellipse. It is always
        returned with respect to the x axis.
        Obtained from: https://en.wikipedia.org/wiki/Jones_calculus#Polarization_axis_from_Jones_vector
        """

        Eox = abs(self.Ex)
        Eoy = abs(self.Ey)
        phix = angle(self.Ex)
        phiy = angle(self.Ey)

        #FIXME: not clear this is right when vetor propagates in -z direction
        x = 2*Eox*Eoy*cos(phix-phiy)
        y = (Eox*Eox-Eoy*Eoy)

        return arctan2(x,y)/2

    def __add__(self, rhs):
        if self.k != rhs.k:
            raise ValueError("JonesVectors can be added when they have the same k")

        # if self.z != rhs.z:
        #     print("Warning: addition of two Jonesvectors from two different z: {0} and {1}".format(self.z, rhs.z))

        return JonesVector(Ex=self.Ex+rhs.Ex, Ey=self.Ey+rhs.Ey, k=self.k, z=self.z)

    def __mul__(self, rightSide):
        if isinstance(rightSide, number_types):
            return self.mul_number(rightSide)
        else:
            raise TypeError(
                "Unrecognized right side element in multiply: '{0}' ({1})\
                 cannot be multiplied by a JonesVector".format(rightSide, type(rightSide)))

    def __rmul__(self, leftSide):
        if isinstance(leftSide, number_types):
            return self.mul_number(leftSide)
        else:
            raise TypeError(
                "Unrecognized left side element in multiply: '{0}'\
                 cannot be multiplied by a JonesVector".format(leftSide))

    def mul_number(self, n):
        """ Multiply a Jones vector by a number."""
        return JonesVector(Ex=self.Ex * n, Ey=self.Ey * n, k=self.k, z=self.z)

    def copy(self):
        return JonesVector(Ex=self.Ex, Ey=self.Ey, k=self.k, z=self.z)

    def reflect(self):
        
        self.b1 = self.b1
        self.b2 = -self.b2
        self.b3 = -self.b3

        # FIXME: Soft reflection? Field stays the same because b2 -> -b2
        # Is this right
        self.E2 = -self.E2

    @property
    def isHorizontallyPolarized(self) -> bool :
        """ The beam is horizontally polarized if Ex != 0 and Ey==0"""
        if isAlmostZero(self.Ey) and abs(self.Ex) != 0:
            return True

        return False

    @property
    def isVerticallyPolarized(self) -> bool :
        """ The beam is vertically polarized if Ey != 0 and Ex==0"""
        if isAlmostZero(self.Ex) and abs(self.Ey) != 0:
            return True

        return False

    @property
    def isLinearlyPolarized(self) -> bool :
        """ The beam is linearly polarized if the phase between both components
        is 0 or pi """
        delta = (angle(self.E1) - angle(self.E2)) % pi

        if isAlmostZero(delta):
            return True

        return False
    
    @property
    def isEllipticallyPolarized(self) -> bool :
        return not (self.isLinearlyPolarized or self.isCircularlyPolarized)

    @property
    def isCircularlyPolarized(self) -> bool :
        return self.isRightCircularlyPolarized or self.isLeftCircularlyPolarized

    @property
    def isRightCircularlyPolarized(self) -> bool :
        """ Returns if the beam is left-circularly polarized.
        Here we take E1 and E2 because they are defined with respect  to
        b1 and b2, and the definition is clear b1 x b2 = b3, therefore the orientation
        for right circular is obtained when E1 is π/2 ahead of E2 (not the other way
        around).

        See definition of RCP and LCP at https://en.wikipedia.org/wiki/Jones_calculus
        """
        if self.E2 == 0:
            return False

        delta = angle(self.E1 / self.E2)
        if areRelativelyAlmostEqual(delta, pi / 2):
            if areRelativelyAlmostEqual(abs(self.E1), abs(self.E2)):
                return True

        return False

    @property
    def isLeftCircularlyPolarized(self) -> bool :
        """ Returns if the beam is left-circularly polarized.
        Here we take E1 and E2 because they are defined with respect  to
        b1 and b2, and the definition is clear b1 x b2 = b3, therefore the orientation
        for right circular is obtained when E1 is π/2 ahead of E2 (not the other way
        around).
        
        See definition of RCP and LCP at https://en.wikipedia.org/wiki/Jones_calculus
        """
        if self.E2 == 0:
            return False

        delta = angle(self.E1 / self.E2)
        if areRelativelyAlmostEqual(delta, -pi / 2):
            if areRelativelyAlmostEqual(abs(self.E1), abs(self.E2)):
                return True

        return False

    @property
    def intensity(self) -> float:
        return self.S0

    @property
    def S0(self) -> float:
        return (self.Ex * conj(self.Ex)).real + (self.Ey * conj(self.Ey)).real

    @property
    def S1(self) -> float:
        return (self.Ex * conj(self.Ex)).real - (self.Ey * conj(self.Ey)).real

    @property
    def S2(self) -> float:
        return (self.Ex * conj(self.Ey)).real + (self.Ey * conj(self.Ex)).real

    @property
    def S3(self) -> float:
        raise NotImplementedError("Check math for Stokes vector S3 component")
        
    @property
    def StokesVector(self) -> (float, float, float, float):
        return (self.S0, self.S1, self.S2, self.S3)

    def __str__(self):
        description = ""
        if isEssentiallyReal(self.Ex):
            description += "Ex = {0:.2f}, ".format(abs(self.Ex))
        elif areRelativelyAlmostEqual(abs(self.Ex), 1.0):
            description += "Ex = exp({0}j), ".format(angleInPiFraction(self.Ex))
        else:
            description += "Ex = {0:.2f} ⨉ exp({1}j), ".format(abs(self.Ex), angleInPiFraction(self.Ex))
        
        if isEssentiallyReal(self.Ey):
            description += "Ey = {0:.2f}".format(abs(self.Ey))
        elif areRelativelyAlmostEqual(abs(self.Ey), 1.0):
            description += "Ey = exp({0}j)".format(angleInPiFraction(self.Ey))
        else:
            description += "Ey = {0:.2f} ⨉ exp({1}j)".format(abs(self.Ey), angleInPiFraction(self.Ey))

        return description

    def physicalField(self, phase=0):
        """ The actual physical field that can be measured in the lab (not the complex one)"""
        xComp = self.b1.x * self.Ex * exp(1j*phase) + self.b2.x * self.Ey * exp(1j*phase)
        yComp = self.b1.y * self.Ex * exp(1j*phase) + self.b2.y * self.Ey * exp(1j*phase)
        return (xComp.real, yComp.real)

    def realField(self, phase=0):
        """ Synonym of physical field """
        return self.physicalField(phase=phase)

    def fullCycle(self):
        """ A list of points representing the electric field during one complete
        cycle. This is used to display the electric field in self.show() """

        cycle=[]
        for i in range(100):
            phi = 2*pi*(i/99)
            cycle.append( self.physicalField(phase=phi) ) 
        return cycle

    def show(self, filename=None): # pragma: no cover
        """Animate the electric field on a plot. The arrow represents the electric
        field at any given time. The dashed line is the complete revolution 
        during one cycle. If a filename is provided, it will be saved."""

        cycle = self.fullCycle()
        x,y = zip(*cycle)

        fig, ax = plt.subplots()
        ax.set_ylim(-1,1)
        ax.set_xlim(-1,1)
        ax.set_aspect(1)

        ax.plot(x,y,'k--')
        ax.plot(0,0,'go',markersize=12)

        def animate(point):
            if len(ax.patches) > 0:
                ax.patches.pop(0)
            patch = plt.Arrow(0,0, point[0], point[1], width=0.1, color='k' )
            ax.add_patch(patch)

            return patch,

        ani = animation.FuncAnimation( fig, animate, frames=cycle, interval=30)
        
        if filename is not None:
            ani.save(filename)

        plt.show()

    @classmethod
    def at(cls, theta, inDegrees=False):
        if inDegrees:
            return JonesVector(cos(theta*radPerDeg), sin(theta*radPerDeg))
        return JonesVector(cos(theta), sin(theta))

    @classmethod
    def horizontal(cls):
        return JonesVector.at(theta=0)

    @classmethod
    def vertical(cls):
        return JonesVector.at(theta=pi/2)

    @classmethod
    def plus45(cls):
        return JonesVector.at(theta=pi/4)

    @classmethod
    def minus45(cls):
        return JonesVector.at(theta=-pi/4)

    @classmethod
    def rightCircular(cls):
        return JonesVector(1, exp(-1j*pi/2)).normalize()

    @classmethod
    def leftCircular(cls):
        return JonesVector(1, exp(1j*pi/2)).normalize()
