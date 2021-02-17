from numpy import complex, exp, array, sqrt, cos, sin, arctan2, pi, conj, abs, angle
from .utils import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class JonesVector:    
    def __init__(self, Ex: complex = 0.0, Ey: complex = 0.0):
        """ The phase of the field is (k*z-omega*t+phi).
        A positive phase phi is a delayed field, and a negative
        phase is an advanced field.

        See https://en.wikipedia.org/wiki/Jones_calculus

        """
        self.Ex = complex(Ex)
        self.Ey = complex(Ey)
        self.z = 0
        self.b1 = array([1,0]) # x̂
        self.b2 = array([0,1]) # ŷ

    def normalize(self):
        """ Normalize the field amplitudes to obtain an intensity of 1 """

        fieldAmplitude = sqrt(self.intensity)
        if fieldAmplitude != 0:
            self.Ex /= fieldAmplitude
            self.Ey /= fieldAmplitude
        return self

    @property
    def orientation(self):
        """ Orientation of the polarization ellipse.
        Obtained from: https://en.wikipedia.org/wiki/Jones_calculus#Polarization_axis_from_Jones_vector
        """
        Eox = abs(self.Ex)
        Eoy = abs(self.Ey)
        phix = angle(self.Ex)
        phiy = angle(self.Ey)

        x = 2*Eox*Eoy*cos(phix-phiy)
        y = (Eox*Eox-Eoy*Eoy)

        return arctan2(x,y)/2

    @property
    def isLinearlyPolarized(self) -> bool :
        """ The beam is linearly polarized if the phase between both components
        is 0 or pi """
        delta = (angle(self.Ex) - angle(self.Ey)) % pi

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
        # See definition of RCP and LCP at https://en.wikipedia.org/wiki/Jones_calculus
        if self.Ey == 0:
            return False

        delta = angle(self.Ex / self.Ey)
        if areRelativelyAlmostEqual(delta, pi / 2):
            if areRelativelyAlmostEqual(abs(self.Ex), abs(self.Ey)):
                return True

        return False

    @property
    def isLeftCircularlyPolarized(self) -> bool :
        if self.Ey == 0:
            return False

        delta = angle(self.Ex / self.Ey)
        if areRelativelyAlmostEqual(delta, -pi / 2):
            if areRelativelyAlmostEqual(abs(self.Ex), abs(self.Ey)):
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
            description += "Ex = {0:.1f}, ".format(abs(self.Ex))
        elif areRelativelyAlmostEqual(abs(self.Ex), 1.0):
            description += "Ex = exp({0}j), ".format(angleInPiMultiple(self.Ex))
        else:
            description += "Ex = {0:.1f} ⨉ exp({1}j), ".format(abs(self.Ex), angleInPiMultiple(self.Ex))
        
        if isEssentiallyReal(self.Ey):
            description += "Ey = {0:.1f}, ".format(abs(self.Ey))
        elif areRelativelyAlmostEqual(abs(self.Ey), 1.0):
            description += "Ey = exp({0}j), ".format(angleInPiMultiple(self.Ey))
        else:
            description += "Ey = {0:.1f} ⨉ exp({1}j), ".format(abs(self.Ey), angleInPiMultiple(self.Ey))

        return description

    def fullCycle(self):
        """ A list of points representing the electric field during one complete
        cycle. This is used to display the electric field in self.show() """

        cycle=[]
        j = complex(0,1)
        for i in range(100):
            phi = 2*pi*(i/99)
            complexCoordinates = self.b1 * self.Ex * exp(1j*phi) + self.b2 * self.Ey * exp(1j*phi)
            point = (complexCoordinates[0].real, complexCoordinates[1].real)
            cycle.append( point ) 
        return cycle

    def show(self, filename=None):
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
    def horizontal(cls):
        return JonesVector(1, 0)

    @classmethod
    def vertical(cls):
        return JonesVector(0, 1)

    @classmethod
    def plus45(cls):
        return JonesVector(1, 1).normalize()

    @classmethod
    def minus45(cls):
        return JonesVector(1, -1).normalize()

    @classmethod
    def rightCircular(cls):
        return JonesVector(1, exp(-1j*pi/2)).normalize()

    @classmethod
    def leftCircular(cls):
        return JonesVector(1, exp(1j*pi/2)).normalize()
      