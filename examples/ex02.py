TITLE       = "Predefined polarization states using Jones Vector"
DESCRIPTION = """Many predefined vectors exist such as
JonesVector.vertical()
JonesVector.horizontal()
JonesVector.plus45()
JonesVector.minus45()
JonesVector.rightCircular()
JonesVector.leftCircular()
here we show two rightCircular and leftCircular polarization states"""



from polarization import *

def exampleCode(comments=None):
    print("\nJonesVector.rightCircular()")
    v1 = JonesVector.rightCircular()
    print("Components are {0}".format(v1))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v1.orientation,v1.orientation*degPerRad))
    v1.show()
    print("-----------------------------")
    print("\nJonesVector.leftCircular()")
    v2 = JonesVector.leftCircular()
    print("Components are {0}".format(v2))
    print("Orientation is {0:.2f} rad or {1:.1f}°".format(v2.orientation,v2.orientation*degPerRad))
    v2.show()

if __name__ == "__main__":
    exampleCode()

