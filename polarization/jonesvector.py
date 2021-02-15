import numpy as np
from .utils import *

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
        self.xHat = (1, 0, 0)  # We may rotate the coordinate system
        self.yHat = (0, 1, 0)  # We may rotate the coordinate system
        self.zHat = (0, 0, 1)  # We may rotate the coordinate system

    def normalize(self):
        fieldAmplitude = np.sqrt(self.intensity)
        if fieldAmplitude != 0:
            self.Ex /= fieldAmplitude
            self.Ey /= fieldAmplitude
        return self

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
