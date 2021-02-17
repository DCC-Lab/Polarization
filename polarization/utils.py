import math
import warnings
import inspect
import sys
from numpy import pi, angle

""" Two constants: deg and rad to quickly convert to degrees
or radians with angle*degPerRad or angle*radPerDeg """

degPerRad = 180.0 / math.pi
radPerDeg = math.pi / 180.0


def isAlmostZero(value, epsilon=1e-3):
    return abs(value) < epsilon


def isNotZero(value, epsilon=1e-3):
    return abs(value) > epsilon


def areAbsolutelyAlmostEqual(left, right, epsilon=1e-3):
    return abs(left - right) < epsilon


def areRelativelyAlmostEqual(left, right, epsilon=1e-3):
    absDiff = abs(left - right)
    relTol1 = absDiff / abs(left)
    relTol2 = absDiff / abs(right)
    return relTol1 < epsilon or relTol2 < epsilon


def areAbsolutelyNotEqual(left, right, epsilon=1e-3):
    return abs(left - right) > epsilon


def areRelativelyNotEqual(left, right, epsilon=1e-3):
    return not areRelativelyAlmostEqual(left, right, epsilon)

def isEssentiallyReal(value, epsilon=1e-8) :
    if isAlmostZero(value.imag, epsilon=epsilon):
        return True

    return False

def isEssentiallyImaginary(value, epsilon=1e-8):
    if isAlmostZero(value.real, epsilon=epsilon):
        return True

    return False

def angleInPiMultiple(value):
    theta = angle(value)
    if isAlmostZero(theta):
        return 0
    elif areRelativelyAlmostEqual(theta, pi/2):
        return "π/2"
    elif areRelativelyAlmostEqual(theta, pi/3):
        return "π/3"
    elif areRelativelyAlmostEqual(theta, pi/4):
        return "π/4"
    elif areRelativelyAlmostEqual(theta, pi):
        return "π"
    elif areRelativelyAlmostEqual(theta, 2*pi/3):
        return "2π/3"
    elif areRelativelyAlmostEqual(theta, 3*pi/4):
        return "3π/4"
    elif areRelativelyAlmostEqual(theta, 3*pi/2):
        return "3π/2"
    elif areRelativelyAlmostEqual(theta, 5*pi/4):
        return "5π/4"
    elif areRelativelyAlmostEqual(theta, -pi/2):
        return "-π/2"
    elif areRelativelyAlmostEqual(theta, -pi/3):
        return "-π/3"
    elif areRelativelyAlmostEqual(theta, -pi/4):
        return "-π/4"
    elif areRelativelyAlmostEqual(theta, -pi):
        return "-π"
    elif areRelativelyAlmostEqual(theta, -2*pi/3):
        return "-2π/3"
    elif areRelativelyAlmostEqual(theta, -3*pi/4):
        return "-3π/4"
    elif areRelativelyAlmostEqual(theta, -3*pi/2):
        return "-3π/2"
    elif areRelativelyAlmostEqual(theta, -5*pi/4):
        return "-5π/4"
    return "{0:.2f}".format(theta)

def deprecated(reason: str):
    def deprecatedFunc(func):
        def wrapper(*args, **kwargs):
            warnings.warn(reason, DeprecationWarning)
            return func(*args, **kwargs)

        return wrapper

    return deprecatedFunc


def printClassHierarchy(aClass):
    def printAllChilds(aClass):
        for child in aClass.__subclasses__():
            print("\"{0}\" -> \"{1}\"".format(aClass.__name__, child.__name__))
            printAllChilds(child)
    print("# Paste this in the text field of http://www.graphviz.org")
    print("digraph G {")
    print("  rankdir=\"LR\";")
    printAllChilds(aClass)
    print("}")

def printModuleClasses(moduleName):
    for name, obj in inspect.getmembers(sys.modules[moduleName]):
        if inspect.isclass(obj) and obj.__module__.startswith(moduleName):
            print(obj)
