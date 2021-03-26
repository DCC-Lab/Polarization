from polarization import JonesVector
import numpy as np


class Pulse:
    def __init__(self, lambdaCenter=1.3, lambdaBandwidth=0.13, polarizationState: JonesVector = None, resolution=512):
        self.lambdaCenter = lambdaCenter
        self.lambdaBandwidth = lambdaBandwidth
        self.resolution = resolution
        self.state = polarizationState
        self.kStates = [self.state] * resolution

    @property
    def kc(self):
        return 2 * np.pi / self.lambdaCenter

    @property
    def Dk(self):
        return 2 * np.pi / self.lambdaCenter ** 2 * self.lambdaBandwidth

    @property
    def kSpectrum(self):
        return np.arange(self.kc - self.Dk / 2, self.kc + self.Dk / 2,
                         ((self.kc + self.Dk / 2) - (self.kc - self.Dk / 2)) / self.resolution)
