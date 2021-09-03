TITLE       = " Polarization transformation with waveplates and phase retarders "
DESCRIPTION = """ Vectors can be transformed by JonesMatrices. Any matrix can be created with JonesMatrix(A,B,C,D), 
but there are many predefined matrices, including waveplates and phase retarders:
HWP(theta=pi/4) : halfwave plate at 45°
QWP(theta=pi/4) : quarterwave plate at 45°
PhaseRetarder(): arbitrary retarder
"""



from polarization import *

def exampleCode(comments=None):

    print("\n\nTransform the JonesVector with JonesMatrices")
    print("============================================")
    print("horizontal vector going through quarter waveplate")

    vIn = JonesVector.horizontal()
    v = QWP(theta=pi/4)*vIn

    print("Input components are {0}".format(vIn))
    print("Output components are {0}".format(v))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation,v.orientation*degPerRad))
    print("isCircular {0}".format(v.isCircularlyPolarized))
    v.show()

    print("\n\nApply several JonesMatrices sequentially")
    print("============================================")

    vIn = JonesVector.horizontal()
    v = HWP(theta=pi/2)*QWP(theta=pi/3)*vIn

    print("Input components are {0}".format(vIn))
    print("Output components are {0}".format(v))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation,v.orientation*degPerRad))
    print("v.isLinearlyPolarized        = {0}".format(v.isLinearlyPolarized))
    print("v.isEllipticallyPolarized    = {0}".format(v.isEllipticallyPolarized))
    print("v.isCircularlyPolarized      = {0}".format(v.isCircularlyPolarized))
    print("v.isRightCircularlyPolarized = {0}".format(v.isRightCircularlyPolarized))
    print("v.isLeftCircularlyPolarized  = {0}".format(v.isLeftCircularlyPolarized))
    v.show()

    print("\n\nTransform the JonesVector with an arbitrary phase retarder")
    print("============================================")

    vIn = JonesVector.rightCircular()
    v = PhaseRetarder(None, pi/3, pi/9)*vIn

    print("Input components are {0}".format(vIn))
    print("Output components are {0}".format(v))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation,v.orientation*degPerRad))
    print("v.isLinearlyPolarized        = {0}".format(v.isLinearlyPolarized))
    print("v.isEllipticallyPolarized    = {0}".format(v.isEllipticallyPolarized))
    print("v.isCircularlyPolarized      = {0}".format(v.isCircularlyPolarized))
    print("v.isRightCircularlyPolarized = {0}".format(v.isRightCircularlyPolarized))
    print("v.isLeftCircularlyPolarized  = {0}".format(v.isLeftCircularlyPolarized))
    v.show()

if __name__ == "__main__":
    exampleCode()

