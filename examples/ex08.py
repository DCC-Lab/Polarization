TITLE       = " Polarization transformation after diattenuator "
DESCRIPTION = """ 
linear diattenuator does nor work when a cicular polarized light passes through it??
"""

from polarization import *

def exampleCode(comments=None):

    print("\n\nTransform the JonesVector with diattenuator")
    print("============================================")
    print("horizontal vector going through the diattenuator along the x axis")

    vIn = JonesVector.horizontal()
    v = Diattenuator(2, 0, physicalLength=2) * vIn

    print("Input components are {0}".format(vIn))
    print("Output components are {0}".format(v))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation,v.orientation*degPerRad))
    print("isCircular {0}".format(v.isCircularlyPolarized))
    print("isLinear {0}".format(v.isLinearlyPolarized))
    v.show()

    print("============================================")
    print("right Circular vector going through the diattenuator along the y axis")

    vIn1 = JonesVector.rightCircular()
    v1 = Diattenuator(1, 1, physicalLength=2) * vIn1

    print("Input components are {0}".format(vIn1))
    print("Output components are {0}".format(v1))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v1.orientation,v1.orientation*degPerRad))
    print("isCircular {0}".format(v1.isCircularlyPolarized))
    print("isLinear {0}".format(v1.isLinearlyPolarized))
    v1.show()


if __name__ == "__main__":
    exampleCode()
