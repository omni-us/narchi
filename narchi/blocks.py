
import enum
from .propagators.concat import ConcatenatePropagator
from .propagators.conv import ConvPropagator, PoolPropagator
from .propagators.fixed import AddFixedPropagator, FixedOutputPropagator
from .propagators.group import SequentialPropagator, GroupPropagator
from .propagators.reshape import ReshapePropagator
from .propagators.rnn import RnnPropagator
from .propagators.same import SameShapePropagator, SameShapesPropagator
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

    Identity = SameShapePropagator('Identity')
    """Block that does nothing, useful to connect one tensor to multiple blocks in a graph."""

    Add = SameShapesPropagator('Add')
    """Block that adds the values of all input tensors. Input tensors must have the same shape."""


class ConcatBlocksEnum(enum.Enum):
    """Enum of blocks that concatenate multiple inputs."""

    Concatenate = ConcatenatePropagator('Concatenate')
    """Block that concatenates multiple inputs of the same shape along a given dimension."""


class FixedOutputBlocksEnum(enum.Enum):
    """Enum of blocks that have fixed outputs."""

    Linear = FixedOutputPropagator('Linear')
    """Linear transformation to the last dimension of input tensor."""

    Embedding = AddFixedPropagator('Embedding')
    """A lookup table that retrieves embeddings of a fixed size."""

    AdaptiveAvgPool1d = FixedOutputPropagator('AdaptiveAvgPool1d', unfixed_dims=1, fixed_dims=1)
    """1D adaptive average pooling over input."""

    AdaptiveAvgPool2d = FixedOutputPropagator('AdaptiveAvgPool2d', unfixed_dims=1, fixed_dims=2)
    """2D adaptive average pooling over input."""


class ConvBlocksEnum(enum.Enum):
    """Enum of convolution-style blocks."""

    Conv1d = ConvPropagator('Conv1d', conv_dims=1)
    """1D convolution."""

    Conv2d = ConvPropagator('Conv2d', conv_dims=2)
    """2D convolution."""

    Conv3d = ConvPropagator('Conv3d', conv_dims=3)
    """3D convolution."""

    MaxPool1d = PoolPropagator('MaxPool1d', conv_dims=1)
    """1D maximum pooling."""

    MaxPool2d = PoolPropagator('MaxPool2d', conv_dims=2)
    """2D maximum pooling."""

    MaxPool3d = PoolPropagator('MaxPool3d', conv_dims=3)
    """3D maximum pooling."""

    AvgPool1d = PoolPropagator('AvgPool1d', conv_dims=1)
    """1D average pooling."""

    AvgPool2d = PoolPropagator('AvgPool2d', conv_dims=2)
    """2D average pooling."""

    AvgPool3d = PoolPropagator('AvgPool3d', conv_dims=3)
    """3D average pooling."""


class RnnBlocksEnum(enum.Enum):
    """Enum of recurrent-style blocks."""

    RNN = RnnPropagator('RNN')
    """A simple recurrent block."""

    LSTM = RnnPropagator('LSTM')
    """An LSTM recurrent block."""

    GRU = RnnPropagator('GRU')
    """A GRU recurrent block."""


class ReshapeBlocksEnum(enum.Enum):
    """Enum of blocks that transform the shape."""

    Reshape = ReshapePropagator('Reshape')
    """Transformation of the shape of the input."""


class GroupPropagatorsEnum(enum.Enum):
    """Enum of blocks that group other blocks."""

    Sequential = SequentialPropagator('Sequential')
    """Sequence of blocks that are connected in the given order."""

    Group = GroupPropagator('Group')
    """Group of blocks with connected according to a given graph."""

    Module = ModulePropagator('Module')
    """Definition of a complete module."""


known_propagators = [
    SameShapeBlocksEnum,
    ConcatBlocksEnum,
    FixedOutputBlocksEnum,
    ConvBlocksEnum,
    RnnBlocksEnum,
    ReshapeBlocksEnum,
    GroupPropagatorsEnum,
]


propagators = {}


def register_propagator(propagator, replace=False):
    """Adds a propagator to the dictionary of registered propagators."""
    if not replace and propagator.block_class in propagators:
        raise ValueError('Propagator for blocks of type '+propagator.block_class+' already registered.')
    invalid_classes = {'Default', 'Input', 'Output', 'Nested', 'Shared'}
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
