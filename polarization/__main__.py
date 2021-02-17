from polarization import *

# Define the vector with Ex and Ey
print("\n\nDefine horizontal polarization with JonesVector(Ex=1, Ey=0)")
print("===========================================================")
v = JonesVector(Ex=1, Ey=0) # horizontal
print("Components are {0}".format(v))
print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation,v.orientation*degPerRad))
v.show()
#v.show("horizontal.mp4") #to save movie

print("\n\nDefine other polarizations with JonesVector(Ex, Ey),\ncan be normalized with .normalize()")
print("====================================================")
v = JonesVector(Ex=1, Ey=1).normalize() # +45°
print("Components are {0}".format(v))
print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation,v.orientation*degPerRad))
v.show()
#v.show("plus45.mp4") #to save movie

print("\n\nMany predefined vectors exist")
print("=============================")
# Many predefined vectors:
# JonesVector.vertical()
# JonesVector.horizontal()
# JonesVector.plus45()
# JonesVector.minus45()
# JonesVector.rightCircular()
# JonesVector.leftCircular()

print("\nJonesVector.rightCircular()")
print("-----------------------------")
v = JonesVector.rightCircular() 
print("Components are {0}".format(v))
print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation,v.orientation*degPerRad))
v.show()

print("\nJonesVector.leftCircular()")
print("-----------------------------")
v = JonesVector.leftCircular() # Many predefined vectors
print("Components are {0}".format(v))
print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation,v.orientation*degPerRad))
v.show()

# Arbitrary polarization
print("\n\nArbitrary polarization JonesVector(Ex=1*exp(1j*0.3), Ey=0.5).normalize()")
print("========================================================================")
v = JonesVector(Ex=1*exp(1j*0.3), Ey=0.5).normalize()
print("Components are {0}".format(v))
print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation,v.orientation*degPerRad))
v.show()

# Many properties are defined:
print("\n\nMany properties to access polarization state")
print("============================================")
print("For vector {0}".format(v))
print("v.orientation (0=horizontal) = {0}".format(v.orientation))
print("v.isLinearlyPolarized        = {0}".format(v.isLinearlyPolarized))
print("v.isEllipticallyPolarized    = {0}".format(v.isEllipticallyPolarized))
print("v.isCircularlyPolarized      = {0}".format(v.isCircularlyPolarized))
print("v.isRightCircularlyPolarized = {0}".format(v.isRightCircularlyPolarized))
print("v.isLeftCircularlyPolarized  = {0}".format(v.isLeftCircularlyPolarized))

# Vectors can be transformed by JonesMatrices
# Any matrix can be created with JonesMatrix(A,B,C,D)
# but there are many predefined matrices:
#
# HorizontalPolarizer(): polarizer at theta=0°
# VerticalPolarizer(): polarizer at theta=90° 
# Plus45Polarizer(): polarizer at theta=45°
# Minus45Polarizer(): polarizer at theta=-45°
# HWP(theta=pi/4) : halfwave plate at 45°
# QWP(theta=pi/4) : quarterwave plate at 45° 
# RightCircularPolarizer(): right circular polarizer
# LeftCircularPolarizer(): left circular polarizer
# PhaseRetarder(): arbitrary retarder

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

print("Malus law example")
print("=================")
x = []
y = []
vIn = JonesVector.horizontal()
for theta in range(0,190,10):
    pol = LinearPolarizer(theta=theta*radPerDeg)
    vOut = pol*vIn

    x.append(theta)
    y.append(vOut.intensity)

plt.title("\n\nMalus Law: horizontal beam intensity after linear polarizer")
plt.xlabel(r"Polarizer orientation $\theta$ from horizontal [°]")
plt.ylabel(r"Intensity [arb. unit]")
plt.plot(x,y,'ko')
plt.show()


print(r"""
# Example code for polarization module
# ====================================
from polarization import *

# Define the vector with Ex and Ey
print("\n\nDefine horizontal polarization with JonesVector(Ex=1, Ey=0)")
print("===========================================================")
v = JonesVector(Ex=1, Ey=0) # horizontal
print("Components are {0}".format(v))
print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation,v.orientation*degPerRad))
v.show()
#v.show("horizontal.mp4") #to save movie

print("\n\nDefine other polarizations with JonesVector(Ex, Ey),\ncan be normalized with .normalize()")
print("====================================================")
v = JonesVector(Ex=1, Ey=1).normalize() # +45°
print("Components are {0}".format(v))
print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation,v.orientation*degPerRad))
v.show()
#v.show("plus45.mp4") #to save movie

print("\n\nMany predefined vectors exist")
print("=============================")
# Many predefined vectors:
# JonesVector.vertical()
# JonesVector.horizontal()
# JonesVector.plus45()
# JonesVector.minus45()
# JonesVector.rightCircular()
# JonesVector.leftCircular()

print("\nJonesVector.rightCircular()")
print("-----------------------------")
v = JonesVector.rightCircular() 
print("Components are {0}".format(v))
print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation,v.orientation*degPerRad))
v.show()

print("\nJonesVector.leftCircular()")
print("-----------------------------")
v = JonesVector.leftCircular() # Many predefined vectors
print("Components are {0}".format(v))
print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation,v.orientation*degPerRad))
v.show()

# Arbitrary polarization
print("\n\nArbitrary polarization JonesVector(Ex=1*exp(1j*0.3), Ey=0.5).normalize()")
print("========================================================================")
v = JonesVector(Ex=1*exp(1j*0.3), Ey=0.5).normalize()
print("Components are {0}".format(v))
print("Orientation is {0:.2f} rad or {1:.1f}°".format(v.orientation,v.orientation*degPerRad))
v.show()

# Many properties are defined:
print("\n\nMany properties to access polarization state")
print("============================================")
print("For vector {0}".format(v))
print("v.orientation (0=horizontal) = {0}".format(v.orientation))
print("v.isLinearlyPolarized        = {0}".format(v.isLinearlyPolarized))
print("v.isEllipticallyPolarized    = {0}".format(v.isEllipticallyPolarized))
print("v.isCircularlyPolarized      = {0}".format(v.isCircularlyPolarized))
print("v.isRightCircularlyPolarized = {0}".format(v.isRightCircularlyPolarized))
print("v.isLeftCircularlyPolarized  = {0}".format(v.isLeftCircularlyPolarized))

# Vectors can be transformed by JonesMatrices
# Any matrix can be created with JonesMatrix(A,B,C,D)
# but there are many predefined matrices:
#
# HorizontalPolarizer(): polarizer at theta=0°
# VerticalPolarizer(): polarizer at theta=90° 
# Plus45Polarizer(): polarizer at theta=45°
# Minus45Polarizer(): polarizer at theta=-45°
# HWP(theta=pi/4) : halfwave plate at 45°
# QWP(theta=pi/4) : quarterwave plate at 45° 
# RightCircularPolarizer(): right circular polarizer
# LeftCircularPolarizer(): left circular polarizer
# PhaseRetarder(): arbitrary retarder

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
""")