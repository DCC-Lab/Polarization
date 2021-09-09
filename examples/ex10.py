TITLE       = " User-defined JonesMatrices and JonesVectors "
DESCRIPTION = """ 

"""

from polarization import *

def exampleCode(comments=None):
    print("\n\nHorizontal light passing through a User-defined JonesMatrix")
    v = JonesVector(Ex=1, Ey=0)  # horizontal
    userdefinedMatrix = JonesMatrix(1, 0, 0.5, 1, physicalLength=2)
    v1 = userdefinedMatrix * v
    print("Input components are {0}".format(v))
    print("Output components are {0}".format(v1))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v1.orientation,v1.orientation*degPerRad))
    print("\n\nThe linear polarizer is now oriented at 90 degrees (pi/2)")
    print("isCircular {0}".format(v.isCircularlyPolarized))
    print("isLinear {0}".format(v.isLinearlyPolarized))
    v1.show()

    print("============================================")
    print("\n\nUser-defined JonesVector passing through the LinearPolarizer")
    unpolarizedLight = JonesVector(Ex=1*exp(1j*0.3), Ey=0.5).normalize()
    v2 = LinearPolarizer(0) * unpolarizedLight
    print("Input components are {0}".format(unpolarizedLight))
    print("Output components are {0}".format(v2))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v2.orientation, v2.orientation * degPerRad))
    print("\n\nThe linear polarizer is now oriented at 90 degrees (pi/2)")
    print("isCircular {0}".format(v2.isCircularlyPolarized))
    print("isLinear {0}".format(v2.isLinearlyPolarized))
    v2.show()

if __name__ == "__main__":
    exampleCode()