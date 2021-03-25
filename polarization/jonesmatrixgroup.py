from .jonesvector import JonesVector
# from .jonesmatrix import JonesMatrix
from .utils import number_types
from typing import List
import collections


# class JonesMatrixGroup:
#     def __init__(self, matrices=None):
#         self.matrices = []
#         if matrices is not None:
#             if not isinstance(matrices, collections.Iterable):
#                 raise TypeError("'matrices' must be iterable (i.e. a list or a tuple of JonesMatrix objects).")

#             for matrix in matrices:
#                 self.append(matrix)

#     def append(self, matrix: JonesMatrix):
#         if not isinstance(matrix, JonesMatrix):
#             raise TypeError("'matrix' must be a JonesMatrix instance.")

#         self.matrices.append(matrix)

#     def __len__(self):
#         return len(self.matrices)

#     def __getitem__(self, i):
#         return self.matrices[i]

#     def subGroup(self, startAt: int = 0, stopAt: int = None):
#         return JonesMatrixGroup(matrices=self.matrices[startAt:stopAt])

#     def __mul__(self, rightSide):
#         """ Multiplies a JonesMatrixGroup with another JonesMatrixGroup, a single JonesMatrix,
#         a number or a list of numbers. """
#         if isinstance(rightSide, JonesMatrixGroup):
#             return self.mul_matrixGroup(rightSide)
#         elif isinstance(rightSide, JonesMatrix):
#             return self.mul_matrix(rightSide)
#         elif isinstance(rightSide, JonesVector):
#             return self.mul_vector(rightSide)
#         elif isinstance(rightSide, number_types):
#             return self.mul_number(rightSide)
#         elif isinstance(rightSide, (list, tuple)):
#             if isinstance(rightSide[0], number_types):
#                 return self.mul_number_array(rightSide)
#             elif isinstance(rightSide[0], JonesVector):
#                 return self.mul_vector_array(rightSide)
#         raise TypeError(
#             "Unrecognized right side element in multiply: '{0}'\
#              cannot be multiplied by a JonesMatrixGroup".format(rightSide))

#     def __rmul__(self, leftSide):
#         """ Multiplies a JonesMatrixGroup by a number or a list of numbers. """
#         if isinstance(leftSide, number_types):
#             return self.mul_number(leftSide)
#         elif isinstance(leftSide, (list, tuple)):
#             assert isinstance(leftSide[0], number_types)
#             return self.mul_number_array(leftSide)
#         else:
#             raise TypeError(
#                 "Unrecognized left side element in multiply: '{0}'\
#                  cannot be multiplied with a JonesMatrixGroup".format(leftSide))

#     def __truediv__(self, other):
#         """ Divides a JonesMatrixGroup by a number or a list of numbers. """
#         if isinstance(other, number_types):
#             return self.mul_number(1/other)
#         elif isinstance(other, (list, tuple)):
#             assert isinstance(other[0], number_types)
#             return self.mul_number_array([1/n for n in other])
#         else:
#             raise TypeError(
#                 "Unrecognized left side element in true divide: '{0}'\
#                  cannot be multiplied with a JonesMatrixGroup".format(other))

#     def mul_matrixGroup(self, matrixGroup):
#         assert len(self.matrices) == len(matrixGroup)
#         return JonesMatrixGroup(matrices=[m * n for (m, n) in zip(self.matrices, matrixGroup)])

#     def mul_matrix(self, matrix):
#         return JonesMatrixGroup(matrices=[m * matrix for m in self.matrices])

#     def mul_vector(self, vector) -> List[JonesVector]:
#         """ Returns a list of output vectors.
#             Use self.forwardPropagation instead to propagate on its transfer matrix.
#         """
#         return [m * vector for m in self.matrices]

#     def mul_number(self, n):
#         return JonesMatrixGroup(matrices=[m * n for m in self.matrices])

#     def mul_number_array(self, A):
#         """ Multiply each matrix of the group by the array, element-wise.
#             If its an array of JonesVector, return a list of output JonesVector
#         """
#         assert len(self.matrices) == len(A)
#         return JonesMatrixGroup(matrices=[m * n for (m, n) in zip(self.matrices, A)])

#     def mul_vector_array(self, vectors) -> List[JonesVector]:
#         """ Returns a list of output JonesVector by multiplying each matrix with the corresponding vector. """
#         assert len(self.matrices) == len(vectors)
#         return [m * v for (m, v) in zip(self.matrices, vectors)]

#     @property
#     def shape(self) -> tuple:
#         return len(self.matrices), 2, 2


# class PauliBasis(JonesMatrixGroup):
#     def __init__(self):
#         e0 = JonesMatrix(1, 0, 0, 1)
#         e1 = JonesMatrix(1, 0, 0, -1)
#         e2 = JonesMatrix(0, 1, 1, 0)
#         e3 = JonesMatrix(0, -1j, 1j, 0)

#         super(PauliBasis, self).__init__(matrices=[e0, e1, e2, e3])
