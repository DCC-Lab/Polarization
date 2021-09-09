TITLE       = " Linear polarizer at arbitrary angle "
DESCRIPTION = """ In this example, we assess the polarization state of an input horizontal polarized light after 
passing through a linear polarizer at different angles. Note that the angle of rotation is defined in respect to the
horizontal (x) axis.
 TO DO: CHANGE NUMBER OF EXAMPLE AND PUT IT BETWEEN 2 AND 3"""

from polarization import *

def exampleCode(comments=None):
    print("\n\nDefine horizontal polarization and a linear polarizer oriented at 0 degrees")
    v = JonesVector.horizontal()
    v1 = LinearPolarizer(0) * v
    print("Input components are {0}".format(v))
    print("Output components are {0}".format(v1))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v1.orientation,v1.orientation*degPerRad))
    v1.show()
    print("\n\nThe linear polarizer is now oriented at 90 degrees (pi/2)")

    v1 = LinearPolarizer(pi/2) * v
    print("Input components are {0}".format(v))
    print("Output components are {0}".format(v1))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v1.orientation,v1.orientation*degPerRad))
    v1.show()

    print("\n\nThe linear polarizer can be oriented at any angle, here pi/9 rad")
    v1 = LinearPolarizer(pi/9) * v
    print("Input components are {0}".format(v))
    print("Output components are {0}".format(v1))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v1.orientation,v1.orientation*degPerRad))
    v1.show()

    HorizontalPolarizer().showPolarizationPlot()


if __name__ == "__main__":
    exampleCode()