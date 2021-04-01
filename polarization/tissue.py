from .tissueStack import TissueStack
import numpy as np


class Tissue:
    def __init__(self, shape):
        self.map = np.zeros(shape)  # todo: move map to Map object (and Line)
        self.layers = []

    def stackAt(self, coordinates) -> TissueStack:
        pass

    def display(self):
        """ Display all layer stacks and their properties. """
        pass


class RandomTissue2D(Tissue):
    def __init__(self, width=100, height=3000, surfaceLayer=True, maxBirefringence=0.0042):
        super(RandomTissue2D, self).__init__(shape=(width, height))

        self.surfaceLayer = surfaceLayer
        self.width = width
        self.height = height
        self.maxBirefringence = maxBirefringence
        self.nLayers = np.random.randint(5, 7)
        self.layerHeightRange = (60, 400)

        self.generateMap()
        self.generateLayers()

    def generateMap(self):
        baseOffset = RandomSinusGroup(maxA=10, minF=0.001, maxF=0.1, n=40)
        layerOffsets = []
        for _ in range(self.nLayers):
            layerOffsets.append(RandomSinusGroup(maxA=2, minF=0.01, maxF=0.1, n=5))

        initialOffset = np.random.randint(200, 800)
        initialLengths = np.random.randint(self.layerHeightRange[0], self.layerHeightRange[1], self.nLayers)

        # todo: matrix refactor
        for i in range(self.width):
            lineLengths = [initialOffset + int(baseOffset.eval(i))]
            for j in range(len(initialLengths)):
                lineLengths.append(initialLengths[j] + int(layerOffsets[j].eval(i)))

            # oop: self.map[i].fill(lengths) or map.fill() with matrix refactor
            d = 0
            for ref, L in enumerate(lineLengths):
                self.map[i][d: d+L] = ref
                d += L
            if d < self.height:
                self.map[i][d:] = len(lineLengths) + 1

    def generateLayers(self):
        # generate random layer properties for each layer in the map
        # generate reference TissueStack. Change thickness and reset scatterers when copied
        pass


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
