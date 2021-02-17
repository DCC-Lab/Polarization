# Polarization

By Daniel Côté dccote@cervo.ulaval.ca

*Polarization* is a simple module to manage optical polarization with Jones and Stokes vectors. You create a `JonesVector` or use on of the pre-defined ones, then you apply your  `JonesMatrix` or you use one of the many pre-defined ones. You can visualize the field with `JonesVector.show()`

You need `matplotlib`, which is a fairly standard Python module and `numpy`. If you do not have it, installing [Anaconda](https://www.anaconda.com/download/) is your best option. Python 3.6 or later is required. There are several ways to install the module:

1. Simplest `pip install --upgrade polarization`

   1. If you need to install `pip`, download [getpip.py](https://bootstrap.pypa.io/get-pip.py) and run it with `python getpip.py`

2. If you download the [source](https://github.com/DCC-Lab/Polarization/) of the module, then you can type: `python setup.py install`

3. From GitHub, you can get the latest version (including bugs, which are 153% free!) and then type `python setup.py install`

4. If you are completely lost, copying the folder `polarization` (the one that includes `__init__.py`) from the source file into the same directory as your own script will work.

* To see the example code, type `python -m polarization`. The example code is below and will get printed to your console after it has executed.

You can visualize the field with  `v.show()` or `v.show("movie.mp4")` to save it.

<video src="arbitrary.mp4"  loop=true></video>

```python
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


```

