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

def angleInPiFraction(value):
    theta = angle(value)
    denominators = [1,2,3,4,6,8]

    if isAlmostZero(theta):
        return "0"

    for den in denominators:
        for num in range(-den*2, den*2+1,1):
            if areAbsolutelyAlmostEqual(theta, num*pi/den):
                if num == 1:
                    return "π/{0}".format(den)
                elif num == -1:
                    return "-π/{0}".format(den)
                elif float(num)/float(den) == num // den:
                    return "π/{0}".format(num/den)
                else:
                    return "{0}π/{1}".format(num, den)

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
