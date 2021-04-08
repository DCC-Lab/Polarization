from polarization import JonesVector
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

    @classmethod
    def horizontal(cls, centerWavelength=None, wavelengthBandwidth=None, resolution=512):
        return Pulse(centerWavelength=centerWavelength, wavelengthBandwidth=wavelengthBandwidth, resolution=resolution,
                     polarization=JonesVector.horizontal())

    @classmethod
    def vertical(cls, centerWavelength=None, wavelengthBandwidth=None, resolution=512):
        return Pulse(centerWavelength=centerWavelength, wavelengthBandwidth=wavelengthBandwidth, resolution=resolution,
                     polarization=JonesVector.vertical())

    @classmethod
    def plus45(cls, centerWavelength=None, wavelengthBandwidth=None, resolution=512):
        return Pulse(centerWavelength=centerWavelength, wavelengthBandwidth=wavelengthBandwidth, resolution=resolution,
                     polarization=JonesVector.plus45())

    @classmethod
    def minus45(cls, centerWavelength=None, wavelengthBandwidth=None, resolution=512):
        return Pulse(centerWavelength=centerWavelength, wavelengthBandwidth=wavelengthBandwidth, resolution=resolution,
                     polarization=JonesVector.minus45())

    @classmethod
    def rightCircular(cls, centerWavelength=None, wavelengthBandwidth=None, resolution=512):
        return Pulse(centerWavelength=centerWavelength, wavelengthBandwidth=wavelengthBandwidth, resolution=resolution,
                     polarization=JonesVector.rightCircular())

    @classmethod
    def leftCircular(cls, centerWavelength=None, wavelengthBandwidth=None, resolution=512):
        return Pulse(centerWavelength=centerWavelength, wavelengthBandwidth=wavelengthBandwidth, resolution=resolution,
                     polarization=JonesVector.leftCircular())
