from polarization import JonesVector, DifferentialGroupDelay
import matplotlib.pyplot as plt
from typing import List
import numpy as np


class Pulse:
    def __init__(self, vectors=None, centerWavelength=None, wavelengthBandwidth=None,
                 polarization: JonesVector = None, resolution=512):
        """
        A list of JonesVector to define a pulse with a uniform intensity distribution.
        Provide 'vectors' if they are already defined, or 'centerWavelength', 'wavelengthBandwidth' and 'polarization'
         to create a new pulse with a given polarization. This pulse will have as much JonesVectors as 'resolution'.
        """
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

    def __getitem__(self, item):
        return self.vectors[item]

    @property
    def shape(self):
        return (self.__len__(),)

    @property
    def isExpanded(self):
        return False

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
            return getattr(p, name)
        return [self._nestedProperty(name, e) for e in p]

    @property
    def isExpanded(self):
        return True

    def __iter__(self):
        return iter(self.pulses)

    def __len__(self):
        return len(self.pulses)

    def __getitem__(self, item):
        return self.pulses[item]


class PulseCollection:
    """ A collection of multiple polarization input states ('Pulse' objects). """
    def __init__(self, pulses, DGD=None):
        self.pulses = pulses
        self.DGD = DGD

    @property
    def hasDGD(self):
        return self.DGD is not None

    @property
    def k(self):
        k = self.pulses[0].k
        for pulse in self.pulses[1:]:
            if pulse.k != k:
                raise Exception("Undefined 'k' for PulseCollection since its pulses don't have "
                                "the same 'k' distribution. ")
        return k

    @property
    def fringes(self):
        """ Fringes of shape (2xN_states, width, resolution) """
        out = []
        for pulse in self.pulses:
            out.extend([pulse.Ex, pulse.Ey])
        return np.asarray(out)

    @property
    def intensity(self):
        return 10 * np.log10(np.abs(np.fft.fft(self.fringes, axis=-1) ** 2))

    def __iter__(self):
        return iter(self.pulses)

    def __len__(self):
        return len(self.pulses)

    def __getitem__(self, item):
        return self.pulses[item]

    @property
    def isExpanded(self):
        for pulse in self.pulses:
            if pulse.isExpanded:
                return True
        return False

    def display(self, tag=""):
        fig, axes = plt.subplots(1, 4, figsize=(14, 10))
        for i, ax in enumerate(axes):
            ax.imshow(np.transpose(self.intensity[i]), aspect='auto', cmap='gray')
        plt.tight_layout()
        # plt.savefig("test{}.png".format(tag))
        plt.show()

    def save(self, filePath):
        np.save(filePath, self.fringes)

    @classmethod
    def dualInputStates(cls, centerWavelength, wavelengthBandwidth, resolution=512, DGDAmplitude=0.0):
        p1 = Pulse.horizontal(centerWavelength, wavelengthBandwidth, resolution=resolution)
        p2 = Pulse.plus45(centerWavelength, wavelengthBandwidth, resolution=resolution)

        DGD = DifferentialGroupDelay(centerWavelength, amplitude=DGDAmplitude)
        return PulseCollection(pulses=[p1, p2], DGD=DGD)
