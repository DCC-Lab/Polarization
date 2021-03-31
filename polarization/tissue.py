from .tissueStack import TissueStack
import numpy as np


class Tissue:
    def __init__(self, shape):
        self.map = np.zeros(shape)
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
        self.nLayers = np.random.randint(1, 20)

        self.generate()

    def generate(self):
        self.map  # (width, height)
        X = np.arange(1, self.width+1)
        baseOffset = RandomSinusGroup(maxA=10, minF=0.001, maxF=0.1, n=40)
        initialOffset = np.random.randint(200, 800)

        layerOffsets = []
        for _ in range(self.nLayers):
            layerOffsets.append(RandomSinusGroup(maxA=2, minF=0.01, maxF=0.1, n=5))

        newLayers = deepcopy(RandomChunk(nLayers=self.nLayers, surface=self.surfaceLayer, height=self.height,
                                         max_dn=self.maxBirefringence).layers)
        for i in range(self.width):
            for j, layer in enumerate(newLayers[1:]):
                layer.height += int(layerOffsets[j].eval(X[i]))
            newChunk = Chunk(layers=newLayers, offset=initialOffset + int(baseOffset.eval(X[i])), height=self.height)
            self.appendChunk(newChunk)

            chunkCopy = deepcopy(self.chunks[-1])
            chunkCopy.resetScatterers()
            newLayers = chunkCopy.layers


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

    def distort(self, maxPercentage=0.01):
        # not that useful and creates a high frequency shift
        distortFactor = 1 + np.random.uniform(-maxPercentage, maxPercentage)
        for sinus in self.sinusFunctions:
            sinus.A *= distortFactor
            sinus.f *= distortFactor
            sinus.d *= distortFactor


class RandomSinusGroup(SinusGroup):
    def __init__(self, maxA=1, minF=0.05, maxF=0.1, n=5):
        F = np.random.uniform(low=minF, high=maxF, size=n)
        A = np.random.uniform(low=0, high=maxA, size=n)
        D = np.random.uniform(low=0, high=2*np.pi, size=n)

        sinusFunctions = []
        for i in range(n):
            sinusFunctions.append(Sinus(A[i], F[i], D[i]))

        super(RandomSinusGroup, self).__init__(sinusFunctions)
