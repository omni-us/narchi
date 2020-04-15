Neural network architecture definition package
==============================================

This python package provides functionalities for defining neural network
architectures in an implementation independent way. Furthermore, the dimensions
of all the input and output tensors from the network blocks are not required to
be set. Only dimensions of the inputs to the network need to be given these
shapes are propagated through verifying that the connections between the blocks
are compatible. The given dimensions can be variable names, resulting in that
the derived dimensions are mathematical expressions including these variables.
