TITLE       = " Output intensity for different input linear polarization angles using showPolarizationPlot() "
DESCRIPTION = """ In this example, we show the intensity of light at the output of a component (or group of components)
as a function of the input linear polarization angle. The function showPolarizationPlot() automatically displays the
intensity graph over 180 degrees of input linear polarization angle.

"""
from polarization import *

def exampleCode(comments=None):
    print("\n\nOutput intensity for diffrent input polarization states")
    print("============================================")
    print("Linear polarizer at 0 degree")
    LinearPolarizer(0).showPolarizationPlot()

    print("Half waveplate with angle pi/2 combined with a vertical polarizer")
    v = HWP(pi/2) * VerticalPolarizer()
    v.showPolarizationPlot()

if __name__ == "__main__":
    exampleCode()