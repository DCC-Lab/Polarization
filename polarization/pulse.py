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

    def __iter__(self):
        return iter(self.vectors)

    def __len__(self):
        return len(self.vectors)
