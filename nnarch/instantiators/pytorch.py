"""A standard module instantiator for pytorch."""

import torch
from functools import reduce
from collections import OrderedDict
from .common import instantiate_block
from ..module import ModuleArchitecture
from ..propagators.reshape import check_output_shape, norm_output_shape


class BaseModule(torch.nn.Module, ModuleArchitecture):
    """Base class for pytorch modules based on an nnarch architecture."""

    def __init__(self, *args, state_dict=None, **kwargs):
        torch.nn.Module.__init__(self)
        ModuleArchitecture.__init__(self, *args, **kwargs)

        architecture = self.architecture
        blocks = self.blocks

        io_ids = {x._id for x in architecture.inputs+architecture.outputs}

        for block_id in blocks.keys():
            if block_id not in io_ids:
                block = instantiate_block(blocks[block_id], self.blocks_mappings)
                setattr(self, block_id, block)

        self.state_dict_prop = state_dict


    def forward(self, *args, **kwargs):
        """Runs a forward using the architecture's inputs and graph."""
        architecture = self.architecture
        topological_predecessors = self.topological_predecessors
        inputs = {x for x in kwargs.keys()}
        expected_inputs = {x._id for x in architecture.inputs}
        if len(args) != 0:
            raise RuntimeError(type(self).__name__+' expects only keyword arguments.')
        if inputs != expected_inputs:
            raise RuntimeError(type(self).__name__+' got unexpected arguments, given '+str(inputs)+', expected '+str(expected_inputs)+'.')

        device = next(self.parameters()).device
        values = OrderedDict({x: kwargs[x].to(device) for x in inputs})

        out_ids = {x._id for x in architecture.outputs}

        for node, inputs in topological_predecessors.items():
            if node in out_ids:
                values[node] = values[inputs[0]]
                continue
            result = getattr(self, node)(*[values[x] for x in inputs])
            block_mapping = self.blocks_mappings[self.blocks[node]._class]
            if 'out_index' in block_mapping:
                values[node] = result[block_mapping['out_index']]
            else:
                values[node] = result

        if len(architecture.outputs) == 1:
            return values[architecture.outputs[0]._id]
        else:
            return [values[x._id] for x in architecture.outputs]


    @property
    def state_dict_prop(self):
        """The current state dictionary."""
        return super().state_dict()


    @state_dict_prop.setter
    def state_dict_prop(self, state_dict):
        """Replaces the current state dictionary with the one given.

        Args:
            state_dict (dict): State dictionary to set.
        """
        if state_dict is None:
            return
        elif isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        if not isinstance(state_dict, dict):
            raise ValueError('Expected state_dict to be a dictionary.')

        self.load_state_dict(state_dict)


class Reshape(torch.nn.Module):
    """Reshape module that receives as input an nnarch output_shape object."""

    def __init__(self, output_shape):
        super().__init__()
        output_shape = norm_output_shape(output_shape)
        self.idxs = check_output_shape(output_shape)
        self.output_shape = output_shape

    def forward(self, input):
        """Transforms the shape of the input according to the specification in output_shape."""
        in_dims = input.shape[1:]
        idxs = self.idxs
        if len(input.shape) != len(idxs)+1:
            raise RuntimeError(type(self).__name__+' got a tensor with '+str(len(input.shape))+' dimensions but expected '+str(len(idxs)+1)+'.')

        reshaped = input
        if idxs != [x for x in range(len(idxs))]:
            permute = [0] + [x+1 for x in idxs]
            reshaped = reshaped.permute(*permute)

        output_shape = self.output_shape
        if any(isinstance(x, (list, dict)) for x in output_shape):
            reshape = [input.shape[0]]
            for val in output_shape:
                if isinstance(val, int):
                    reshape.append(in_dims[val])
                elif isinstance(val, list):
                    reshape.append(reduce((lambda x, y: x * y), [in_dims[v] for v in val]))
                elif isinstance(val, dict):
                    raise NotImplementedError('TODO: '+str(val))
            reshaped = reshaped.reshape(reshape)

        return reshaped


class Sequential(torch.nn.Sequential):
    """Sequential module that receives as input an nnarch blocks object."""
    def __init__(self, blocks, blocks_mappings):
        subblock_list = []
        for subblock in blocks:
            subblock_list.append(instantiate_block(subblock, blocks_mappings))
        super().__init__(*subblock_list)


class Add(torch.nn.Module):
    """Module that adds all inputs element-wise, thus expects all inputs to be the same shape."""
    def forward(self, *inputs):
        return sum(inputs)


standard_pytorch_blocks_mappings = {
    'Sequential': {
        'class': 'nnarch.instantiators.pytorch.Sequential',
    },
    'Add': {
        'class': 'nnarch.instantiators.pytorch.Add',
    },
    'Reshape': {
        'class': 'nnarch.instantiators.pytorch.Reshape',
    },
    'Conv2d': {
        'class': 'torch.nn.Conv2d',
        'kwargs': {
            'in_channels': 'shape:in:0',
            'out_channels': 'output_size',
        },
    },
    'BatchNorm2d': {
        'class': 'torch.nn.BatchNorm2d',
        'kwargs': {
            'num_features': 'shape:in:0',
        },
    },
    'ReLU': {
        'class': 'torch.nn.ReLU',
    },
    'LeakyReLU': {
        'class': 'torch.nn.LeakyReLU',
    },
    'MaxPool2d': {
        'class': 'torch.nn.MaxPool2d',
    },
    'AdaptiveAvgPool2d': {
        'class': 'torch.nn.AdaptiveAvgPool2d',
    },
    'LSTM': {
        'class': 'torch.nn.LSTM',
        'kwargs': {
            'input_size': 'shape:in:1',
            'batch_first': 'const:bool:True',
            ':skip:': 'output_size',
        },
        'out_index': 0,
    },
    'Identity': {
        'class': 'torch.nn.Identity',
    },
    'Linear': {
        'class': 'torch.nn.Linear',
        'kwargs': {
            'in_features': 'shape:in:-1',
            'out_features': 'output_size',
        },
    },
}


class StandardModule(BaseModule):
    blocks_mappings = standard_pytorch_blocks_mappings
