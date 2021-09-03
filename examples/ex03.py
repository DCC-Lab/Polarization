TITLE       = "Polarization Transformation after different Linear Polarizers"
DESCRIPTION = """Vectors can be transformed by JonesMatrices
Any matrix can be created with JonesMatrix(A,B,C,D)
but there are many predefined matrices for polarizers such as:
HorizontalPolarizer(): polarizer at theta=0°
VerticalPolarizer(): polarizer at theta=90° 
Plus45Polarizer(): polarizer at theta=45°
Minus45Polarizer(): polarizer at theta=-45°
RightCircularPolarizer(): right circular polarizer
LeftCircularPolarizer(): left circular polarizer
Here we assess polarization state of different three different input polarization state
after passing through Horizontal Polarizer based on the JonesVector and JonesMatrices,
the vector can be normalized with .normalize()
"""


from polarization import *

def exampleCode(comments=None):
    print("\n\nTransform the JonesVector with JonesMatrices")
    print("============================================")
    print("horizontal vector going through HorizontalPolarizer")
    vIn1 = JonesVector.horizontal()
    v1 = HorizontalPolarizer() * vIn1
    print("Input components are {0}".format(vIn1))
    print("Output components are {0}".format(v1))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v1.orientation,v1.orientation*degPerRad))
    print("isCircular {0}".format(v1.isCircularlyPolarized))
    print("isLinear {0}".format(v1.isLinearlyPolarized))
    v1.show()
    print("============================================")
    print("linearly polarized vector at 45° going through HorizontalPolarizer")
    vIn2 = JonesVector.plus45()
    v2 = HorizontalPolarizer() * vIn2
    print("Input components are {0}".format(vIn2))
    print("Output components are {0}".format(v2))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v2.orientation, v2.orientation * degPerRad))
    print("isCircular {0}".format(v2.isCircularlyPolarized))
    print("isLinear {0}".format(v2.isLinearlyPolarized))
    v2.show()
    print("============================================")
    print("Circularly polarized vector going through HorizontalPolarizer")
    vIn3 = JonesVector.rightCircular()
    v3 = HorizontalPolarizer() * vIn3
    print("Input components are {0}".format(vIn3))
    print("Output components are {0}".format(v3))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v3.orientation, v3.orientation * degPerRad))
    print("isCircular {0}".format(v3.isCircularlyPolarized))
    print("isLinear {0}".format(v3.isLinearlyPolarized))
    v3.show()

if __name__ == "__main__":
	exampleCode()