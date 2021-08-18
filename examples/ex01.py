TITLE       = "Linear Polarized Light"
DESCRIPTION = """Here we can define differently orientated linear polarized light
 based on the different parameters of the Jones Vectors, the vector can be normalized with .normalize()
 e.g. v = JonesVector(Ex=1, Ey=1).normalize() which is +45° polarized """



from polarization import *

def exampleCode(comments=None):
    print("\n\nDefine horizontal polarization with JonesVector(Ex=1, Ey=2)")
    print("===========================================================")
    v = JonesVector(Ex=1, Ey=0)  # horizontal
    print("Components are {0}".format(v))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation, v.orientation * degPerRad))
    v.show()
    # v.show("horizontal.mp4") #to save movie

if __name__ == "__main__":
	exampleCode()