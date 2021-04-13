from polarization import JonesVector
from typing import List
import numpy as np


class Pulse:
    def __init__(self, vectors=None, centerWavelength=None, wavelengthBandwidth=None,
                 polarization: JonesVector = None, resolution=512):
        self.vectors = []
        if vectors is not None:
            self.vectors = vectors
        elif centerWavelength is None or wavelengthBandwidth is None or polarization is None:
            raise AttributeError("Missing argument to construct the pulse. ")
        else:
            kc = 2 * np.pi / centerWavelength
            Dk = 2 * np.pi / centerWavelength ** 2 * wavelengthBandwidth
            K = np.arange(kc - Dk / 2, kc + Dk / 2, ((kc + Dk / 2) - (kc - Dk / 2)) / resolution)
            for k in K:
                v = polarization.copy()
                v.k = k
                self.vectors.append(v)

    @property
    def k(self) -> list:
        return [v.k for v in self.vectors]

    @property
    def kc(self) -> float:
        return np.mean(self.k)

    @property
    def Ex(self):
        return [v.Ex for v in self.vectors]

    @property
    def Ey(self):
        return [v.Ey for v in self.vectors]

    def display(self):
        # todo: display fringes
        pass

    def __iter__(self):
        return iter(self.vectors)

    def __len__(self):
        return len(self.vectors)

    @property
    def shape(self):
        return (self.__len__(),)

    @classmethod
    def horizontal(cls, centerWavelength, wavelengthBandwidth, resolution=512):
        return Pulse(centerWavelength=centerWavelength, wavelengthBandwidth=wavelengthBandwidth, resolution=resolution,
                     polarization=JonesVector.horizontal())

    @classmethod
    def vertical(cls, centerWavelength, wavelengthBandwidth, resolution=512):
        return Pulse(centerWavelength=centerWavelength, wavelengthBandwidth=wavelengthBandwidth, resolution=resolution,
                     polarization=JonesVector.vertical())

    @classmethod
    def plus45(cls, centerWavelength, wavelengthBandwidth, resolution=512):
        return Pulse(centerWavelength=centerWavelength, wavelengthBandwidth=wavelengthBandwidth, resolution=resolution,
                     polarization=JonesVector.plus45())

    @classmethod
    def minus45(cls, centerWavelength, wavelengthBandwidth, resolution=512):
        return Pulse(centerWavelength=centerWavelength, wavelengthBandwidth=wavelengthBandwidth, resolution=resolution,
                     polarization=JonesVector.minus45())

    @classmethod
    def rightCircular(cls, centerWavelength, wavelengthBandwidth, resolution=512):
        return Pulse(centerWavelength=centerWavelength, wavelengthBandwidth=wavelengthBandwidth, resolution=resolution,
                     polarization=JonesVector.rightCircular())

    @classmethod
    def leftCircular(cls, centerWavelength, wavelengthBandwidth, resolution=512):
        return Pulse(centerWavelength=centerWavelength, wavelengthBandwidth=wavelengthBandwidth, resolution=resolution,
                     polarization=JonesVector.leftCircular())


class PulseArray:
    """ Expanded pulse of shape (B, P) or (C, B, P) for a single input state after scanning a tissue. """
    def __init__(self, pulses: List[Pulse]):
        self.pulses = pulses

    @property
    def shape(self, p=None):
        if p is None:
            p = self.pulses
        if type(p) is not list:
            return []
        return [len(p)] + self.shape(p[0])

    @property
    def Ex(self):
        return self._nestedProperty('Ex')

    @property
    def Ey(self):
        return self._nestedProperty('Ey')

    def _nestedProperty(self, name, p=None):
        if p is None:
            p = self.pulses
        if type(p) is Pulse:
            return [getattr(p, name)]
        return [self._nestedProperty(name, e) for e in p]

