# Questions/comments about Polarization coding

You may be using the Python module for calculations, but you may also want to dig into the code.  It is coded with teaching purpose in mind, so here are some questions/comments about how it was coded and why it was coded like that.

## What is the overall design goal?

* Simplicity of usage: it must provide an API to obtain results for experiments in a clear readable manner with the mathematical details

* Replication of physical meaning by the classes: classes must encapsulate the expected behaviour and properties without relying on external variables

* Readability of the code is more important than performance.

  

## What is the overall design strategy?

* To keep the code simple: we provide an API that focusses on what people will want to calculate, not on the mathematical details to perform the calculation.  The classes defined replicate the physics and the general formalism, but they hide the details.
* Replication of physical meaning by the classes
  * we make no assumption on many aspects (basis vectors, direction of propagation) so we hardcode our assumptions, with the reasoning that when needed, we will implement to the possibility of changing them.  For example, the `JonesVector` class has `b1` and `b2` for the basis of the two fields.  They currently are fixed at hat(x) and hat(y), therefore, the code assume Ex and Ey everywhere.   However, it may change if we find the need to change the basis, at which point `b1` and `b2` will be made settable and the rest of the code will be adapted.
  * the idea is not to let the user perform all the calculations with `JonesMatrices` manually: we have the tools to make the computer do the work.  We do not want to work in place of the computer.
  * For example, a `JonesMatrix` has a property called `birefringence`: it will return the axis of birefringence and the value.  Of course, this is easy for we know the matrix, but it is better to use a general approach that works for any matrices, including a simple matrrix like a phase plate.  The birefringence exists, regardless of how I set it up.  Therefore `birefringence` will diagonalize the matrix and if possible will return the axes and the birefringence, which is the real way of doing it and will work with any combination of matices.
* Use of `camelCase` is what we do.  I think this reads better than `other_options`.  We name variables so that the sentence reads normally: `jonesVector.isLinearlyPolarized` is highly readable and natural.



