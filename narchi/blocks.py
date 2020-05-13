
import enum
from .propagators.conv import ConvPropagator, PoolPropagator
from .propagators.fixed import FixedOutputPropagator
from .propagators.group import SequentialPropagator, GroupPropagator
from .propagators.reshape import ReshapePropagator
from .propagators.rnn import RnnPropagator
from .propagators.same import SameShapePropagator
from .module import ModulePropagator


class SameShapeBlocksEnum(enum.Enum):
    """Enum of blocks that preserve the input shape."""

    Sigmoid = SameShapePropagator('Sigmoid')
    """Block that applies a sigmoid function."""

    LogSigmoid = SameShapePropagator('LogSigmoid')
    """Block that applies a log-sigmoid function."""

    Softmax = SameShapePropagator('Softmax')
    """Block that applies a softmax function."""

    LogSoftmax = SameShapePropagator('LogSoftmax')
    """Block that applies a log-softmax function."""

    Tanh = SameShapePropagator('Tanh')
    """Block that applies a hyperbolic tangent function."""

    ReLU = SameShapePropagator('ReLU')
    """Block that applies a rectified linear unit function."""

    LeakyReLU = SameShapePropagator('LeakyReLU')
    """Block that applies a leaky rectified linear unit function."""

    Dropout = SameShapePropagator('Dropout')
    """Block that applies dropout, randomly set elements to zero."""

    BatchNorm2d = SameShapePropagator('BatchNorm2d')
    """Block that does 2D batch normalization."""

    Add = SameShapePropagator('Add', multi_input=True)
    """Block that adds the values of all input tensors. Input tensors must have the same shape."""

    Identity = SameShapePropagator('Identity')
    """Block that does nothing, useful to connect one input to multiple blocks in a graph."""


class FixedOutputBlocksEnum(enum.Enum):
    """Enum of blocks that have fixed outputs."""

    Linear = FixedOutputPropagator('Linear')
    """Linear transformation to the last dimension of input tensor.

    Shape:

    - Input: (\\*, input_size) where input_size is implied from previous block.
    - Output: (\\*, output_size) where \\* is any number of dimensions with same sizes as input.

    Parameters:

    - output_size (int): Number of output features.
    """

    AdaptiveAvgPool2d = FixedOutputPropagator('AdaptiveAvgPool2d', unfixed_dims=1, fixed_dims=2)
    """2D adaptive average pooling over input.

    Shape:

    - Input: (C, I1, I2) where C is the number of input channels and (I1, I2) are any size.
    - Output: (C, O1, O2) where (O1, O2) is always the same independent of the input size.

    Parameters:

    - output_size ([int, int]): Number of output features (O1, O2).
    """


class ConvBlocksEnum(enum.Enum):
    """Enum of convolution-style blocks."""

    Conv1d = ConvPropagator('Conv1d', conv_dims=1)
    """1D convolution.

    Parameters:

    - output_size (int): Number of filters or output channels.
    - kernel_size (int): The size of the convolution kernel.
    - stride (int): Step size for sliding the kernel over the input. Default 1.
    - padding (int): Padding added to all borders of the input. Default 0.
    """

    Conv2d = ConvPropagator('Conv2d', conv_dims=2)
    """2D convolution.

    Parameters:

    - output_size (int): Number of filters or output channels.
    - kernel_size (int): The size of the convolution kernel.
    - stride (int): Step size for sliding the kernel over the input. Default 1.
    - padding (int): Padding added to all borders of the input. Default 0.
    """

    Conv3d = ConvPropagator('Conv3d', conv_dims=3)
    """3D convolution.

    Parameters:

    - output_size (int): Number of filters or output channels.
    - kernel_size (int): The size of the convolution kernel.
    - stride (int): Step size for sliding the kernel over the input. Default 1.
    - padding (int): Padding added to all borders of the input. Default 0.
    """

    MaxPool1d = PoolPropagator('MaxPool1d', conv_dims=1)
    """1D maximum pooling.

    Parameters:

    - kernel_size (int): The size of the pooling kernel.
    - stride (int): Step size for sliding the kernel over the input. Default 1.
    - padding (int): Padding added to all borders of the input. Default 0.
    """

    MaxPool2d = PoolPropagator('MaxPool2d', conv_dims=2)
    """2D maximum pooling.

    Parameters:

    - kernel_size (int): The size of the pooling kernel.
    - stride (int): Step size for sliding the kernel over the input. Default 1.
    - padding (int): Padding added to all borders of the input. Default 0.
    """

    MaxPool3d = PoolPropagator('MaxPool3d', conv_dims=3)
    """3D maximum pooling.

    Parameters:

    - kernel_size (int): The size of the pooling kernel.
    - stride (int): Step size for sliding the kernel over the input. Default 1.
    - padding (int): Padding added to all borders of the input. Default 0.
    """

    AvgPool1d = PoolPropagator('AvgPool1d', conv_dims=1)
    """1D average pooling.

    Parameters:

    - kernel_size (int): The size of the pooling kernel.
    - stride (int): Step size for sliding the kernel over the input. Default 1.
    - padding (int): Padding added to all borders of the input. Default 0.
    """

    AvgPool2d = PoolPropagator('AvgPool2d', conv_dims=2)
    """2D average pooling.

    Parameters:

    - kernel_size (int): The size of the pooling kernel.
    - stride (int): Step size for sliding the kernel over the input. Default 1.
    - padding (int): Padding added to all borders of the input. Default 0.
    """

    AvgPool3d = PoolPropagator('AvgPool3d', conv_dims=3)
    """3D average pooling.

    Parameters:

    - kernel_size (int): The size of the pooling kernel.
    - stride (int): Step size for sliding the kernel over the input. Default 1.
    - padding (int): Padding added to all borders of the input. Default 0.
    """


class RnnBlocksEnum(enum.Enum):
    """Enum of recurrent-style blocks."""

    RNN = RnnPropagator('RNN')
    """A simple recurrent block.

    Shape:

    - Input: (L, input_size) where L is the length of the sequence and input_size is implied from previous block.
    - Output: (L, output_size) where output_size is given as a parameter.

    Parameters:

    - output_size (int or var): Number of output features.
    """

    LSTM = RnnPropagator('LSTM')
    """An LSTM recurrent block.

    Same as :attr:`narchi.blocks.RnnBlocksEnum.RNN`.
    """

    GRU = RnnPropagator('GRU')
    """A GRU recurrent block.

    Same as :attr:`narchi.blocks.RnnBlocksEnum.RNN`.
    """


class ReshapeBlocksEnum(enum.Enum):
    """Enum of glocks that transform the shape."""

    Reshape = ReshapePropagator('Reshape')
    """Transformation of the shape of the input.

    Shape:

    - Input: Any shape that conforms to the reshape_spec.
    - Output: The transformed shape according to reshape_spec.

    Parameters:

    - reshape_spec: Specification of reshape transformation.
    """


class GroupPropagatorsEnum(enum.Enum):
    """Enum of blocks that group other blocks."""

    Sequential = SequentialPropagator('Sequential')
    """Sequence of blocks that are connected in the give order.

    Shape:

    - Input: As the first block in the sequence.
    - Output: As the last block in the sequence.

    Parameters:

    - blocks (list[dict, ...]): List of blocks included in the sequence.
    """

    Group = GroupPropagator('Group')
    """Group of blocks with connected according to a given graph.

    Shape:

    - Input: As the input block of the group.
    - Output: As the output block of the group.

    Parameters:

    - blocks (list[dict, ...]): List of blocks included in the group.
    - graph (list[str, ...]): List of strings defining connections between blocks.
    - input (str): Identifier of the input block.
    - output (str): Identifier of the output block.
    """

    Module = ModulePropagator('Module')
    """Definition of a complete module."""


known_propagators = [
    SameShapeBlocksEnum,
    FixedOutputBlocksEnum,
    ConvBlocksEnum,
    RnnBlocksEnum,
    ReshapeBlocksEnum,
    GroupPropagatorsEnum]


propagators = {}


def register_propagator(propagator, replace=False):
    """Adds a propagator to the dictionary of registered propagators."""
    if not replace and propagator.block_class in propagators:
        raise ValueError('Propagator for blocks of type '+propagator.block_class+' already registered.')
    invalid_classes = {'Default', 'Input', 'Output', 'Nested'}
    if propagator.block_class in invalid_classes or propagator.block_class.startswith('Nested'):
        raise ValueError('Propagators are not allowed to have as class any of '+str(invalid_classes)+'.')
    propagators[propagator.block_class] = propagator


def register_known_propagators():
    """Function that registers all propagators defined in the modules of the package."""
    for propagators_enum in known_propagators:
        for propagator in propagators_enum:
            assert propagator.name == propagator.value.block_class
            register_propagator(propagator.value)


register_known_propagators()
