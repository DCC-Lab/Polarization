from .tissueStack import *
from .pulse import *
from typing import Union
import numpy as np

__all__ = ['Tissue', 'RandomTissue2D']


class Tissue:
    def __init__(self, stacks: List[TissueStack] = None, height=None, width=None, depth=None):
        """ A collection of TissueStacks to act as a tissue sample. """
        # todo: change shape dims to microns
        # todo: allow 3D tissue
        self.stacks = []
        if stacks:
            self.stacks = stacks
            self.height = stacks[0].height
            self.width = len(self.stacks)
        else:
            self.height = height
            self.width = width
        self.depth = depth

        self._scattDensity = None
        self._opticAxis = None
        self._apparentOpticAxis = None
        self._birefringence = None

    def scan(self, pulse: Union[Pulse, PulseCollection], verbose=False):
        if verbose:
            def v_print(*args, **kwargs):
                return print(*args, **kwargs)
        else:
            v_print = lambda *a, **k: None

        if type(pulse) is Pulse:
            return self._scanPulse(pulse, v_print)
        elif type(pulse) is PulseCollection:
            return self._scanPulseCollection(pulse, v_print)
        else:
            return ValueError("Can only scan Pulse or PulseCollection objects. "
                              "Type {} was provided.".format(type(pulse)))

    def _scanPulse(self, pulse, v_print) -> PulseArray:
        bScan = []
        for b in range(self.width):
            v_print(" .Stack {}/{}".format(b+1, self.width))
            bScan.append(self.stacks[b].backscatterMany(pulse))
        return PulseArray(bScan)

    def _scanPulseCollection(self, pulses, v_print) -> PulseCollection:
        assert not pulses.isExpanded, "Cannot scan a PulseCollection that was already scanned. "

        pulsesBScan = [[] for _ in pulses]
        for a in range(self.width):
            v_print(" .Stack {}/{}".format(a+1, self.width))
            pulsesALine = self.stacks[a].backscatterMany(pulses)
            for p in range(len(pulses)):
                pulsesBScan[p].append(pulsesALine[p])

        return PulseCollection(pulses=[PulseArray(bScan) for bScan in pulsesBScan])

    def __iter__(self):
        return iter(self.stacks)

    def __len__(self):
        return len(self.stacks)

    @property
    def scattDensity(self):
        if self._scattDensity is None:
            self._scattDensity = self.stacks[0].scattDensity[:, None]
            for stack in self.stacks[1:]:
                self._scattDensity = np.concatenate([self._scattDensity, stack.scattDensity[:, None]], axis=1)
        return self._scattDensity

    @property
    def opticAxis(self):
        if self._opticAxis is None:
            self._opticAxis = self.stacks[0].opticAxis[:, :, None]
            for stack in self.stacks[1:]:
                self._opticAxis = np.concatenate([self._opticAxis, stack.opticAxis[:, :, None]], axis=2)
        return self._opticAxis

    @property
    def apparentOpticAxis(self):
        if self._apparentOpticAxis is None:
            self._apparentOpticAxis = self.stacks[0].apparentOpticAxis[:, :, None]
            for stack in self.stacks[1:]:
                self._apparentOpticAxis = np.concatenate([self._apparentOpticAxis, stack.apparentOpticAxis[:, :, None]], axis=2)
        return self._apparentOpticAxis

    @property
    def birefringence(self):
        if self._birefringence is None:
            self._birefringence = self.stacks[0].birefringence[:, None]
            for stack in self.stacks[1:]:
                self._birefringence = np.concatenate([self._birefringence, stack.birefringence[:, None]], axis=1)
        return self._birefringence

    def display(self, title=None):
        """ Display all layer stacks and their properties. """
        fig, axes = plt.subplots(1, 5, figsize=(15, 5), sharey="all", sharex="all")
        self._displayToAxes(axes)

        fig.suptitle(title, fontsize=14)
        plt.show()

    def _displayToAxes(self, axes):
        axes[0].imshow(self.scattDensity, aspect='auto', vmin=0, vmax=20, interpolation='none')
        axes[0].set_title("Scattering Density")
        axes[1].imshow(self.opticAxis[0], aspect='auto', vmin=-1, vmax=1, cmap='bwr', interpolation='none')
        axes[1].set_title("OA1")
        axes[2].imshow(self.opticAxis[1], aspect='auto', vmin=-1, vmax=1, cmap='bwr', interpolation='none')
        axes[2].set_title("OA2")
        axes[3].imshow(self.opticAxis[2], aspect='auto', vmin=-1, vmax=1, cmap='bwr', interpolation='none')
        axes[3].set_title("OA3")
        axes[4].imshow(self.birefringence, aspect='auto', vmin=0, vmax=0.003, cmap='gray', interpolation='none')
        # todo: link vmax to sim layer.max_dn...
        axes[4].set_title("Birefringence")

    def save(self, filePath, reSample: int = None):

        sampleData = np.concatenate([self.scattDensity[None, :, :],
                                     # self.apparentOpticAxis,  # todo
                                     self.opticAxis,
                                     self.birefringence[None, :, :]])

        if reSample is not None:
            picks = np.linspace(0, sampleData.shape[1]-1, num=reSample, dtype=np.int)
            sampleData = np.take(sampleData, picks, axis=1)

        np.save(filePath, np.moveaxis(sampleData, 2, 1))


class RandomTissue2D(Tissue):
    def __init__(self, height=3000, width=200,
                 referenceStack=None, flat=False,
                 surface=False, maxBirefringence=0.0042, nLayers=None, offset=None, layerHeightRange=(60, 400),
                 sinMaxA=50, sinMaxF=0.05):
        """ Generate a 2D Tissue from a given referenceStack or RandomTissueStack properties.
        The generated Tissue will have varying layer thicknesses and positions
        to simulate a real sample, unless flat=True. """
        # todo: move generator-related logic to a new parent class 'TissueGenerator' or 'RandomTissue'

        if referenceStack is None:
            referenceStack = RandomTissueStack(surface=surface, maxBirefringence=maxBirefringence,
                                               nLayers=nLayers, offset=offset, layerHeightRange=layerHeightRange)
        super(RandomTissue2D, self).__init__(stacks=[], height=height, width=width, depth=1)

        self.flat = flat
        self._layerSizeMap = None
        self._sinMaxA = sinMaxA
        self._sinMaxF = sinMaxF
        self.referenceStack: TissueStack = referenceStack

        self.generateMap()
        self.generateStacks()

    @property
    def nLayers(self):
        return len(self.referenceStack)

    def generateMap(self):
        initialLengths = [self.referenceStack.offset, *[layer.thickness for layer in self.referenceStack]]

        if not self.flat:
            offSets = [RandomSinusGroup(maxA=self._sinMaxA, minF=0.001, maxF=self._sinMaxF, n=40)]
            offSets.extend([RandomSinusGroup(maxA=2, minF=0.01, maxF=0.1, n=5) for _ in range(self.nLayers)])

            for i, (L, sinOffset) in enumerate(zip(initialLengths, offSets)):
                self._layerSizeMap[i] = np.array(L + sinOffset(np.arange(self.width)), dtype=int)
        else:
            for i, L in enumerate(initialLengths):
                self._layerSizeMap[i] = np.full(self.width, L, dtype=int)

    @property
    def referenceStack(self):
        return self._referenceStack

    @referenceStack.setter
    def referenceStack(self, stack):
        self._referenceStack = stack
        if stack is not None:
            self._layerSizeMap = np.squeeze(np.zeros((self.width, self.depth)))
            self._layerSizeMap = np.broadcast_to(self._layerSizeMap, (1 + len(stack), *self._layerSizeMap.shape)).copy()
        else:
            self._layerSizeMap = None

    def _stackOf(self, layerSizes):
        layers = []
        currentHeight = layerSizes[0]
        for thickness, layer in zip(layerSizes[1:], self.referenceStack.layers):
            if currentHeight >= self.height:
                continue
            elif currentHeight + thickness > self.height:
                thickness = self.height - currentHeight
            layers.append(layer.copy(thickness=thickness))
            currentHeight += thickness
        stack = TissueStack(offset=layerSizes[0], layers=layers, height=self.height)
        return stack

    def generateStacks(self):
        for w in range(self.width):
            layerSizes = self._layerSizeMap[:, w]
            self.stacks.append(self._stackOf(layerSizes))


class Sinus:
    def __init__(self, A=1, f=1, d=0):
        self.A = A
        self.f = f
        self.d = d

    def __call__(self, x):
        return self.A * np.sin(self.f * x + self.d)


class SinusGroup:
    def __init__(self, sinusFunctions):
        self.sinusFunctions = sinusFunctions

    def __call__(self, x):
        res = 0
        for sinus in self.sinusFunctions:
            res += sinus(x)
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
