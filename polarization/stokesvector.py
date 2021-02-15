import numpy as np
from simulations.utilities import *


class StokesVector:
    def __init__(self, v:JonesVector):
        self.vector = v
        self.z = 0

    @property
    def S0(self):
        return self.vector.S0
    
    @property
    def S1(self):
        return self.vector.S1

    @property
    def S2(self):
        return self.vector.S2

    @property
    def S3(self):
        return self.vector.S3
