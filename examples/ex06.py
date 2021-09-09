TITLE       = "Polarization Transformation after BirefringentMaterial"
DESCRIPTION = """ 
"""


from polarization import *

def exampleCode(comments=None):
    print("\n\nTransform the JonesVector with JonesMatrices")
    print("============================================")
    print("horizontal vector going through BirefringentMaterial")
    vIn1 = JonesVector(Ex=1, Ey=1, wavelength=532)
    v1 = BirefringentMaterial(deltaIndex=0.2, fastAxisOrientation=0, physicalLength=2)* vIn1
    print("Input components are {0}".format(vIn1))
    print("Output components are {0}".format(v1))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v1.orientation,v1.orientation*degPerRad))
    print("isCircular {0}".format(v1.isCircularlyPolarized))
    print("isLinear {0}".format(v1.isLinearlyPolarized))
    v1.show()
    # print("============================================")
    # print("linearly polarized vector at 45° going through BirefringentMaterial")
    # vIn2 = JonesVector.plus45()
    # v2 = BirefringentMaterial() * vIn2
    # print("Input components are {0}".format(vIn2))
    # print("Output components are {0}".format(v2))
    # print("Orientation is {0:.2f} rad or {1:.1f}°".format(v2.orientation, v2.orientation * degPerRad))
    # print("isCircular {0}".format(v2.isCircularlyPolarized))
    # print("isLinear {0}".format(v2.isLinearlyPolarized))
    # v2.show()
    # print("============================================")
    # print("Circularly polarized vector going through BirefringentMaterial")
    # vIn3 = JonesVector.rightCircular()
    # v3 = BirefringentMaterial() * vIn3
    # print("Input components are {0}".format(vIn3))
    # print("Output components are {0}".format(v3))
    # print("Orientation is {0:.2f} rad or {1:.1f}°".format(v3.orientation, v3.orientation * degPerRad))
    # print("isCircular {0}".format(v3.isCircularlyPolarized))
    # print("isLinear {0}".format(v3.isLinearlyPolarized))
    # v3.show()

if __name__ == "__main__":
	exampleCode()