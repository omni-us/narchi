"""A pytorch module instantiator that supports 1d and 2d packed sequences."""

import math
import torch
import numpy as np
from copy import deepcopy
from collections import namedtuple
from torch.nn.utils.rnn import PackedSequence

from .pytorch import BaseModule, Reshape, standard_pytorch_blocks_mappings


class Packed2dSequence(namedtuple('Packed2dSequence', 'data lengths gaps')):
    """Named tuple for packed 2d sequences."""

    def to(self, *args, **kwargs):
        """Performs dtype and/or device conversion on `self.data`."""
        data = self.data.to(*args, **kwargs)
        return type(self)(data=data, lengths=self.lengths, gaps=self.gaps)


def pack_2d_sequences(input, gap_size:int=0, length_fact:int=1, fail_if_unsorted=True):
    """Packs 3-dim tensors into a long 3-dim tensor by concatenating along the last dimension.

    Args:
        input (tuple/list[tensor]): Input tensors to pack with shapes [C, H, Wi].
        gap_size (int): Size for gap between samples.
        length_fact (int): Increase gaps so that length of inputs are multiple of length_fact.
        fail_if_unsorted (bool): Whether to raise ValueError if received unsorted input.

    Returns:
        Packed2dSequence(data=tensor[1, C, H, lengths+gaps], lenghts=list[len(input)], gaps=[len(input)])
    """
    if not (isinstance(input, (tuple, list))
            and all([hasattr(x, 'shape') for x in input])
            and all([x.dim() == 3 for x in input])
            and all([x.shape[0] == input[0].shape[0] for x in input])
            and all([x.shape[1] == input[0].shape[1] for x in input])):
        raise RuntimeError('Input required to be a tuple/list of tensors all with 3 dimensions and '
                           'have the same size for the first two dimensions.')

    ## Check that input is sorted from longest to shortest ##
    lengths = np.array([x.shape[2] for x in input], dtype=int)
    if any([d > 0 for d in np.diff(lengths)]):
        if fail_if_unsorted:
            raise ValueError('Expected input to be sorted from longest to shortest.')
        input = list(input)
        input.sort(key=lambda x: x.shape[2], reverse=True)
        lengths = np.array([x.shape[2] for x in input], dtype=int)

    ## Determine gaps ##
    gaps = np.zeros(len(input), dtype=int)
    for num in range(len(input)):
        gaps[num] = gap_size if num < len(input)-1 else 0
        fact_off = (lengths[num]+gaps[num]) % length_fact
        if fact_off > 0:
            gaps[num] += length_fact - fact_off

    ## Create tensor for packing ##
    total_length = lengths.sum() + gaps.sum()
    packed = torch.zeros((1, *input[0].shape[0:2], total_length), dtype=input[0].dtype, device=input[0].device)  # pylint: disable=no-member

    ## Copy input to packed tensor ##
    offset = 0
    for num in range(len(input)):
        packed[0, :, :, offset:offset+lengths[num]] = input[num]
        offset += lengths[num] + gaps[num]

    return Packed2dSequence(data=packed, lengths=lengths, gaps=gaps)


def packed_2d_set_gaps_to_zero(packed):
    """Sets to zero the gap locations of a Packed2dSequence."""
    assert isinstance(packed, Packed2dSequence), 'Expected input to be a Packed2dSequence.'
    lengths = packed.lengths
    gaps = packed.gaps
    if len(lengths) > 1 and gaps.sum() > 0:
        offset = 0
        for num in range(len(lengths)):
            packed.data[0, :, :, offset+lengths[num]:offset+lengths[num]+gaps[num]] = 0.0
            offset += lengths[num] + gaps[num]
    return packed


def packed_2d_to_1d(packed_2d):
    """Converts a Packed2dSequence to a PackedSequence."""
    assert isinstance(packed_2d, Packed2dSequence), 'Expected input to be a Packed2dSequence.'

    data = packed_2d.data
    lengths = packed_2d.lengths
    gaps = packed_2d.gaps

    ## Reserve memory for packed 1d tensor ##
    packed_1d_length = lengths.sum()
    packed_1d_size = (packed_1d_length, np.prod(data.shape[1:3]))
    data_1d = torch.zeros(packed_1d_size, dtype=data.dtype, device=data.device)  # pylint: disable=no-member

    ## Get packed 2d sequence offsets and batch_sizes ##
    offsets = [0]
    batch_sizes = torch.zeros(lengths[0], dtype=torch.long)  # pylint: disable=no-member
    for num, length in enumerate(lengths):
        offsets.append(offsets[num]+length+gaps[num])
        batch_sizes[0:length] += 1

    ## Copy data to packed 1d tensor ##
    t = 0
    p = 0
    for nsamp in batch_sizes:
        for num in range(nsamp):
            data_1d[p, :] = data[0, :, :, offsets[num]+t].reshape(-1)
            p += 1
        t += 1

    ## Return PackedSequence ##
    sorted_indices = torch.tensor([num for num in range(len(lengths))], dtype=torch.long, device=data.device)  # pylint: disable=not-callable,no-member
    return PackedSequence(data_1d, batch_sizes, sorted_indices, sorted_indices)


class ReshapePacked(Reshape):
    """Reshape module that works with Packed2dSequence."""

    def forward(self, input):
        """Transforms the shape of the input according to the specification in reshape_spec."""
        if not isinstance(input, Packed2dSequence):
            return super().forward(input)
        if self.reshape_spec != [2, [0, 1]]:
            raise RuntimeError(f'Reshape of Packed2dSequence only supported for reshape_spec=[2, [0, 1]].')
        return packed_2d_to_1d(input)


class Conv2dPacked(torch.nn.Conv2d):
    """Extension of torch.nn.Conv2d that works with Packed2dSequence.

    After convolving, the gaps are set to zero to guaranty that these values are
    not considered when computing gradients or subsequent forwards.

    @todo If convolution overlaps between samples, raise an exception.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        if kernel_size//2 != padding:
            raise NotImplementedError('Currently only implemented for kernel_size//2 == padding.')
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            #stride=stride,
            padding=padding,
            #dilation=dilation,
            #groups=groups,
            bias=bias,
            padding_mode='zeros')

    def forward(self, input):
        if not isinstance(input, Packed2dSequence):
            return super().forward(input)
        output_data = super().forward(input.data)
        #_save_image_detached(output_data[0,0:3,:,:], 'post_conv.png')
        output = packed_2d_set_gaps_to_zero(Packed2dSequence(data=output_data, lengths=input.lengths, gaps=input.gaps))
        #_save_image_detached(output.data[0,0:3,:,:], 'post_conv_mask.png')
        return output


class MaxPool2dPacked(torch.nn.MaxPool2d):
    """Extension of torch.nn.MaxPool2d that works with Packed2dSequence.

    The lengths and gaps are re-estimated.
    """

    def __init__(self, kernel_size, stride):
        if kernel_size != stride:
            raise NotImplementedError('Currently only implemented for kernel_size == stride.')
        super().__init__(kernel_size=kernel_size, stride=stride)

    def forward(self, input):
        if not isinstance(input, Packed2dSequence):
            return super().forward(input)

        ## Compute new packed lengths and gaps ##
        lengths = np.zeros(input.lengths.shape, dtype=int)
        gaps = np.zeros(input.gaps.shape, dtype=int)
        for num in range(len(lengths)):
            length_in = input.lengths[num]
            gap_in = input.gaps[num]
            if (length_in+gap_in) % self.kernel_size > 0:
                raise RuntimeError(f'Expected sum of lengths of sample and gap be a multiple of pooling kernel '
                                   f'size: num={num} kernel_size={self.kernel_size} length={length_in} gap={gap_in}.')
            #if self.ceil_mode:
            lengths[num] = math.ceil(length_in/self.kernel_size)
            #else:
            #    lengths[num] = math.floor(length_in/self.kernel_size)
            gaps[num] = math.floor(gap_in/self.kernel_size)

        ## Perform pooling ##
        #_save_image_detached(input.data[0,0:3,:,:], 'pre_maxpool.png')
        output_data = super().forward(input.data)
        #_save_image_detached(output_data[0,0:3,:,:], 'post_maxpool.png')

        ## Check that pooled tensor matches shape ##
        if lengths.sum()+gaps.sum() != output_data.shape[-1]:
            raise RuntimeError('Bug in implementation, pooling length differs w.r.t. lengths and gaps computation.')

        return Packed2dSequence(data=output_data, lengths=lengths, gaps=gaps)


class BatchNorm2dPacked(torch.nn.BatchNorm2d):
    """Extension of torch.nn.BatchNorm2d that works with Packed2dSequence.

    After normalization, the gaps are set to zero to guaranty that these values
    are not considered when computing gradients or subsequent forwards.
    """
    def forward(self, input):
        if not isinstance(input, Packed2dSequence):
            return super().forward(input)
        output_data = super().forward(input.data)
        #_save_image_detached(output_data[0,0:3,:,:], 'post_bnorm.png')
        output = packed_2d_set_gaps_to_zero(Packed2dSequence(data=output_data, lengths=input.lengths, gaps=input.gaps))
        #_save_image_detached(output.data[0,0:3,:,:], 'post_bnorm_mask.png')
        return output


class LeakyReLU2dPacked(torch.nn.LeakyReLU):
    """Extension of torch.nn.LeakyReLU that works with Packed2dSequence."""
    def forward(self, input):
        if not isinstance(input, Packed2dSequence):
            return super().forward(input)
        output_data = super().forward(input.data)
        return Packed2dSequence(data=output_data, lengths=input.lengths, gaps=input.gaps)


class Linear1dPacked(torch.nn.Linear):
    """Extension of torch.nn.Linear that works with PackedSequence."""
    def forward(self, input):
        if not isinstance(input, PackedSequence):
            return super().forward(input)
        output_data = super().forward(input.data)
        return PackedSequence(data=output_data, batch_sizes=input.batch_sizes, sorted_indices=input.sorted_indices)


mappings = {
    'Reshape': {
        'class': 'narchi.instantiators.pytorch_packed.ReshapePacked',
    },
    'Conv2d': {
        'class': 'narchi.instantiators.pytorch_packed.Conv2dPacked',
        'kwargs': {
            'in_channels': 'shape:in:0',
            'out_channels': 'output_feats',
        },
    },
    'MaxPool2d': {
        'class': 'narchi.instantiators.pytorch_packed.MaxPool2dPacked',
    },
    'BatchNorm2d': {
        'class': 'narchi.instantiators.pytorch_packed.BatchNorm2dPacked',
        'kwargs': {
            'num_features': 'shape:in:0',
        },
    },
    'LeakyReLU': {
        'class': 'narchi.instantiators.pytorch_packed.LeakyReLU2dPacked',
    },
    'Linear': {
        'class': 'narchi.instantiators.pytorch_packed.Linear1dPacked',
        'kwargs': {
            'in_features': 'shape:in:-1',
            'out_features': 'output_feats',
        },
    },
}

packed_pytorch_blocks_mappings = deepcopy(standard_pytorch_blocks_mappings)
packed_pytorch_blocks_mappings.update(mappings)


class PackedModule(BaseModule):
    blocks_mappings = packed_pytorch_blocks_mappings

    def __init__(self, *args, gap_size:int=0, length_fact:int=1, **kwargs):
        """Initializer for PackedModule class.

        Args:
            gap_size (int): Size for gap between samples.
            length_fact (int): Increase gaps so that length of inputs are multiple of length_fact.
            args/kwargs: All other arguments accepted by :class:`.BaseModule`.
        """
        BaseModule.__init__(self, *args, **kwargs)
        self.gap_size = gap_size
        self.length_fact = length_fact


    def inputs_preprocess(self, values):
        """Converts tuples of tensors into Packed2dSequence.

        Args:
            values (OrderedDict): Inputs to the module.
        """
        for key, value in values.items():
            if isinstance(value, (tuple, list)):
                values[key] = pack_2d_sequences(value, gap_size=self.gap_size, length_fact=self.length_fact)


#def _save_image_detached(data, file_path):
#    """Saves a tensor as an image, but only after cloning it and detaching it so that it does not interfere with current computations.
#
#    Args:
#        data (torch.Tensor): Tensor with a shape supported by torchvision.utils.save_image.
#        file_path (str): Path to location where to save the image with a valid image extension.
#    """
#    save_image(data.clone().detach().requires_grad_(False), file_path)
