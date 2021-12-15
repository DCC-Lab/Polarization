from polarization import *


wavelength = 1.3
bandwidth = 0.13
resolution = 256

np.random.seed(111)

dk = 2 * np.pi / wavelength ** 2 * bandwidth
dz = np.pi / dk
height = int(np.pi / dk * resolution)

tissue = RandomTissue2D(width=20, nLayers=3, height=height, maxBirefringence=0.003)
tissue.display()

pIn = PulseCollection.dualInputStates(wavelength, bandwidth, resolution=resolution)
pOut = tissue.scan(pIn, verbose=True)

pOut.display()
