import numpy as np
from numpy import complex, exp, array, sqrt, pi
from .utils import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class JonesVector:    
    def __init__(self, Ex: np.complex = 0.0, Ey: np.complex = 0.0):
        """ The phase of the field is (k*z-omega*t+phi).
        A positive phase phi is a delayed field, and a negative
        phase is an advanced field.

        See https://en.wikipedia.org/wiki/Jones_calculus

        """
        self.Ex = np.complex(Ex)
        self.Ey = np.complex(Ey)
        self.z = 0
        self.e1 = np.array([1, 0])  # We may rotate the coordinate system
        self.e2 = np.array([0, 1])  # We may rotate the coordinate system

    def normalize(self):
        fieldAmplitude = np.sqrt(self.intensity)
        if fieldAmplitude != 0:
            self.Ex /= fieldAmplitude
            self.Ey /= fieldAmplitude
        return self

    def animate(self):
        cycle = self.fullCycle()
        x,y = zip(*cycle)

        fig, ax = plt.subplots()
        ax.set_ylim(-1,1)
        ax.set_xlim(-1,1)
        ax.set_aspect(1)

        ax.plot(x,y,'k')
        line, = ax.plot(0,0,'ko',markersize=12)

        def animate(point):
            line.set_xdata(point[0])
            line.set_ydata(point[1])

        ani = animation.FuncAnimation( fig, animate, frames=cycle, interval=30)
        plt.show()        

    def fullCycle(self):
        cycle=[]
        j = complex(0,1)
        for i in range(100):
            phi = 2*pi*(i/99)
            complexCoordinates = self.e1 * self.Ex * exp(1j*phi) + self.e2 * self.Ey * exp(1j*phi)
            point = (complexCoordinates[0].real, complexCoordinates[1].real)
            cycle.append( point ) 
        return cycle

    @property
    def isLinearlyPolarized(self) -> np.bool :
        # angle returns phase within -pi to pi.
        delta = (np.angle(self.Ex) - np.angle(self.Ey)) % np.pi

        if isAlmostZero(delta):
            return True

        return False
    
    @property
    def isEllipticallyPolarized(self) -> np.bool :
        return not (self.isLinearlyPolarized or self.isCircularlyPolarized)

    @property
    def isCircularlyPolarized(self) -> np.bool :
        return self.isRightCircularlyPolarized or self.isLeftCircularlyPolarized

    @property
    def isRightCircularlyPolarized(self) -> np.bool :
        # See definition of RCP and LCP at https://en.wikipedia.org/wiki/Jones_calculus
        if self.Ey == 0:
            return False

        delta = np.angle(self.Ex / self.Ey)
        if areRelativelyAlmostEqual(delta, np.pi / 2):
            if areRelativelyAlmostEqual(np.abs(self.Ex), np.abs(self.Ey)):
                return True

        return False

    @property
    def isLeftCircularlyPolarized(self) -> np.bool :
        if self.Ey == 0:
            return False

        delta = np.angle(self.Ex / self.Ey)
        if areRelativelyAlmostEqual(delta, -np.pi / 2):
            if areRelativelyAlmostEqual(np.abs(self.Ex), np.abs(self.Ey)):
                return True

        return False

    @property
    def intensity(self) -> float:
        return self.S0

    @property
    def S0(self) -> float:
        return (self.Ex * np.conj(self.Ex)).real + (self.Ey * np.conj(self.Ey)).real

    @property
    def S1(self) -> float:
        return (self.Ex * np.conj(self.Ex)).real - (self.Ey * np.conj(self.Ey)).real

    @property
    def S2(self) -> float:
        return (self.Ex * np.conj(self.Ey)).real + (self.Ey * np.conj(self.Ex)).real

    @property
    def S3(self) -> float:
        raise NotImplementedError("Check math for Stokes vector S3 component")
        
    @property
    def StokesVector(self) -> (np.float, np.float, np.float, np.float):
        return (self.S0, self.S1, self.S2, self.S3)

    def __str__(self):
        return "({0},{1}) at z={2}".format(self.Ex, self.Ey, self.z)
