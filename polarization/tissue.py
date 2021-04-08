from .tissueStack import *
import numpy as np

__all__ = ['Tissue', 'RandomTissue2D']


class Tissue:
    def __init__(self, referenceStack=None, height=3000, width=200, depth=1):
        # todo: change shape dims to microns
        self.height = height
        self.width = width
        self.depth = depth
        self.map = None

        self.referenceStack = referenceStack

    @property
    def referenceStack(self):
        return self._referenceStack

    @referenceStack.setter
    def referenceStack(self, stack):
        self._referenceStack = stack
        if stack is not None:
            self.map = np.squeeze(np.zeros((self.width, self.depth)))
            self.map = np.broadcast_to(self.map, (1 + self.nLayers, *self.map.shape)).copy()
        else:
            self.map = None

    @property
    def nLayers(self):
        return len(self.referenceStack)

    def generateMap(self):
        # overwrite
        pass

    def stackAt(self, coordinates) -> TissueStack:
        """ Tissue Stack of an A-Line at a specified location. """
        if type(coordinates) is int:
            line = self.map[:, coordinates]
        elif len(coordinates) == 2:
            line = self.map[:, coordinates[0], coordinates[1]]
        else:
            raise ValueError

        layers = []
        for L, layer in zip(line[1:], self.referenceStack):
            layer.thickness = L
            layers.append(layer)

        return TissueStack(offset=line[0], layers=layers)

    def display(self):
        """ Display all layer stacks and their properties. """
        pass


class RandomTissue2D(Tissue):
    def __init__(self, height=3000, width=200,
                 referenceStack=None,
                 surface=False, maxBirefringence=0.0042, nLayers=None, offset=None, layerHeightRange=(60, 400)):

        if referenceStack is None:
            referenceStack = RandomTissueStack(surface=surface, maxBirefringence=maxBirefringence,
                                               nLayers=nLayers, offset=offset, layerHeightRange=layerHeightRange)

        super(RandomTissue2D, self).__init__(referenceStack=referenceStack, height=height, width=width, depth=1)

        self.generateMap()

    def generateMap(self):
        offSets = [RandomSinusGroup(maxA=10, minF=0.001, maxF=0.1, n=40)]
        offSets.extend([RandomSinusGroup(maxA=2, minF=0.01, maxF=0.1, n=5) for _ in range(self.nLayers)])
        initialLengths = [self.referenceStack.offset, *[layer.thickness for layer in self.referenceStack]]

        for i, (L, dL) in enumerate(zip(initialLengths, offSets)):
            self.map[i] = np.array(L + dL.eval(np.arange(self.width)), dtype=int)


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
