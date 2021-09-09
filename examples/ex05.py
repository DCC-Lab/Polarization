TITLE       = " Rotation of Jones matrices "
DESCRIPTION = """ In this example, we apply a rotation over a JonesMatrix. The function rotatedBy() can be applied to
 any JonesMatrix. In this case, we use a horizontal polarized light as an input to show the effect of rotating a 
 horizontal polarizer to 90 degrees and 180 degrees."""

from polarization import *

def exampleCode(comments=None):
    print("\n\nRotate Jones Matrices")
    print("============================================")
    print("Horizontal vector going through horizontal polarizer")
    vIn = JonesVector.horizontal()
    pol = HorizontalPolarizer()
    v = pol * vIn
    print("Input components are {0}".format(vIn))
    print("Output components are {0}".format(v))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation,v.orientation*degPerRad))
    v.show()

    print("Horizontal vector going through the horizontal polarizer rotated by 90 degrees")
    v = pol.rotatedBy(pi / 2) * vIn
    print("Input components are {0}".format(vIn))
    print("Output components are {0}".format(v))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation, v.orientation * degPerRad))
    v.show()

    print("Horizontal vector going through the horizontal polarizer rotated by 180 degrees")
    v = pol.rotatedBy(pi) * vIn
    print("Input components are {0}".format(vIn))
    print("Output components are {0}".format(v))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation, v.orientation * degPerRad))
    v.show()

if __name__ == "__main__":
    exampleCode()
