from .tissueStack import *
import numpy as np


class Tissue:
    def __init__(self, referenceStack=None, height=3000, width=200, depth=1):
        # todo: change shape dims to microns
        self.height = height
        self.referenceStack = referenceStack
        self.map = np.zeros((None, width, depth))


    def generateMap(self):
        # overwrite
        pass

    def generateLayers(self):
        self.layers = RandomTissueStack().layers

    def stackAt(self, coordinates) -> TissueStack:
        pass

    def display(self):
        """ Display all layer stacks and their properties. """
        pass


class RandomTissue2D(Tissue):
    def __init__(self, height=3000, width=200,
                 referenceStack=None,
                 surface=True, maxBirefringence=0.0042, nLayers=None, offset=None, layerHeightRange=(60, 400)):

        if referenceStack is None:
            referenceStack = RandomTissueStack(surface=surface, maxBirefringence=maxBirefringence,
                                               nLayers=nLayers, offset=offset, layerHeightRange=layerHeightRange)

        super(RandomTissue2D, self).__init__(referenceStack=referenceStack, height=height, width=width, depth=1)

        self.generateMap()

    def generateMap(self):
        offSets = [RandomSinusGroup(maxA=10, minF=0.001, maxF=0.1, n=40)]
        offSets.extend([RandomSinusGroup(maxA=2, minF=0.01, maxF=0.1, n=5) for _ in range(self.nLayers)])
        initialLengths = [self.referenceStack.offset, *[layer.thickness for layer in self.referenceStack.layers]]

        # fixme: what is nLayers (does it includes empty regions) and how do we initialize map shape
        for i, (L, dL) in enumerate(zip(initialLengths, offSets)):
            self.map[i] = np.array(L + dL.eval(np.arange(self.width)), dtype=int)
        self.map[-1] = self.height - np.sum(self.map, axis=0)


class Sinus:
    def __init__(self, A=1, f=1, d=0):
        self.A = A
        self.f = f
        self.d = d

    def eval(self, x):
        return self.A * np.sin(self.f * x + self.d)


class SinusGroup:
    def __init__(self, sinusFunctions):
        self.sinusFunctions = sinusFunctions

    def eval(self, x):
        res = 0
        for sinus in self.sinusFunctions:
            res += sinus.eval(x)
        return res


class RandomSinusGroup(SinusGroup):
    def __init__(self, maxA=1, minF=0.05, maxF=0.1, n=5):
        F = np.random.uniform(low=minF, high=maxF, size=n)
        A = np.random.uniform(low=0, high=maxA, size=n)
        D = np.random.uniform(low=0, high=2*np.pi, size=n)

        sinusFunctions = []
        for i in range(n):
            sinusFunctions.append(Sinus(A[i], F[i], D[i]))

        super(RandomSinusGroup, self).__init__(sinusFunctions)
