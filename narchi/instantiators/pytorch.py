"""A standard module instantiator for pytorch."""

import torch
from functools import reduce
from collections import OrderedDict
from jsonargparse import Path, get_config_read_mode
from typing import List

from .common import instantiate_block, id_strip_parent_prefix
from ..module import ModuleArchitecture
from ..propagators.reshape import check_reshape_spec, norm_reshape_spec
from ..propagators.group import get_blocks_dict
from ..graph import parse_graph
from ..schemas import auto_tag


class BaseModule(ModuleArchitecture, torch.nn.Module):
    """Base class for instantiation of pytorch modules based on a narchi architecture."""

    def __init__(
        self,
        *args,
        state_dict: dict = None,
        debug: bool = False,
        **kwargs
    ):
        """Initializer for BaseModule class.

        Args:
            state_dict: State dictionary to set.
            debug: Enable to keep self.intermediate_outputs.
            args/kwargs: All other arguments accepted by :class:`.ModuleArchitecture`.
        """
        torch.nn.Module.__init__(self)
        ModuleArchitecture.__init__(self, *args, **kwargs)

        architecture = self.architecture
        blocks = self.blocks
        module_cfg = {'propagated': True, 'cwd': self.cfg.cwd}
        if 'cfg' in kwargs and 'ext_vars' in kwargs['cfg']:
            module_cfg['ext_vars'] = kwargs['cfg']['ext_vars']

        io_ids = {x._id for x in architecture.inputs+architecture.outputs}

        for block_id in blocks.keys():
            if block_id not in io_ids:
                block = instantiate_block(blocks[block_id], self.blocks_mappings, module_cfg)
                setattr(self, id_strip_parent_prefix(block_id), block)

        self.state_dict_prop = state_dict
        self.debug = debug


    def forward(self, *args, **kwargs):
        """Runs a forward using the architecture's inputs and graph."""
        inputs = {x for x in kwargs.keys()}
        expected_inputs = {x._id for x in self.architecture.inputs}
        if len(args) != 0:
            raise RuntimeError(f'{type(self).__name__} expects only keyword arguments.')
        if inputs != expected_inputs:
            raise RuntimeError(f'{type(self).__name__} got unexpected arguments, given {inputs}, expected {expected_inputs}.')

        values = OrderedDict({x: kwargs[x] for x in inputs})
        self.inputs_preprocess(values)
        self.check_inputs_shape(values)
        device = next(self.parameters()).device
        for key, value in values.items():
            values[key] = value.to(device)

        out_ids = {x._id for x in self.architecture.outputs}
        graph_forward(self, values, out_ids, intermediate_outputs=self.debug)

        if len(self.architecture.outputs) == 1:
            return values[self.architecture.outputs[0]._id]
        else:
            return [values[x._id] for x in self.architecture.outputs]


    def inputs_preprocess(self, values):
        """Pre-processing for inputs, base implementation does nothing."""
        pass


    def check_inputs_shape(self, values):
        """Raises error if the shape of any input does not agree with architecture."""
        for node in self.architecture.inputs:
            given = self.get_tensor_shape(values[node._id])
            expected = node._shape
            if len(expected) != len(given) \
               or any(isinstance(expected[n], int) and given[n] != expected[n] for n in range(len(given))):
                raise RuntimeError(f'{type(self).__name__} got unexpected tensor shape for input "{node._id}", '
                                   f'given {given}, expected {expected}.')


    def get_tensor_shape(self, value) -> List[int]:
        """Returns tensor shape excluding batch dimension."""
        return list(value.shape[1:])


    @property
    def state_dict_prop(self):
        """The current state dictionary."""
        return super().state_dict()


    @state_dict_prop.setter
    def state_dict_prop(self, state_dict: dict):
        """Replaces the current state dictionary with the one given.

        Args:
            state_dict: State dictionary to set.
        """
        if state_dict is None:
            return
        elif isinstance(state_dict, (str, Path)):
            path = Path(state_dict, mode=get_config_read_mode(), cwd=self.cfg.cwd)
            state_dict = torch.load(path(), weights_only=True)
        if not isinstance(state_dict, dict):
            raise ValueError('Expected state_dict to be a dictionary.')

        self.load_state_dict(state_dict)


class Sequential(torch.nn.Sequential):
    """Sequential module that receives as input an narchi blocks object."""
    def __init__(self, blocks, blocks_mappings, module_cfg):
        subblock_list = []
        for subblock in blocks:
            subblock_list.append(instantiate_block(subblock, blocks_mappings, module_cfg))
        super().__init__(*subblock_list)


class Add(torch.nn.Module):
    """Module that adds all inputs element-wise, thus expects all inputs to be the same shape."""
    def forward(self, *inputs):
        return sum(inputs)


class Concatenate(torch.nn.Module):
    """Module that concatenates input tensors along a given dimension."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim if dim < 0 else dim+1

    def forward(self, *inputs):
        return torch.cat(inputs, self.dim)  # pylint: disable=no-member


class BaseRNN:
    """Methods for RNN classes to allow disabling the return of the hidden and cell states."""
    def __init__(self, *args, output_state=False, **kwargs):
        self.output_state = output_state
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = super().forward(input)
        return output if self.output_state else output[0]


class RNN(BaseRNN, torch.nn.RNN):
    """Extension of torch.nn.RNN that allows disabling the return of the hidden and cell states."""


class GRU(BaseRNN, torch.nn.GRU):
    """Extension of torch.nn.GRU that allows disabling the return of the hidden and cell states."""


class LSTM(BaseRNN, torch.nn.LSTM):
    """Extension of torch.nn.LSTM that allows disabling the return of the hidden and cell states."""


class Reshape(torch.nn.Module):
    """Reshape module configurable by a narchi reshape_spec object."""

    def __init__(self, reshape_spec):
        super().__init__()
        self.reshape_spec = norm_reshape_spec(reshape_spec)
        self.idxs = check_reshape_spec(self.reshape_spec)

    def forward(self, input):
        """Transforms the shape of the input according to the specification in reshape_spec."""
        in_dims = input.shape[1:]
        idxs = self.idxs
        if self.reshape_spec == 'flatten':
            idxs = [n for n in range(len(input.shape)-1)]
        if len(input.shape) != len(idxs)+1:
            raise RuntimeError(f'{type(self).__name__} got a tensor with {len(input.shape)} dimensions but expected {len(idxs)+1}.')

        reshaped = input
        if idxs != [x for x in range(len(idxs))]:
            permute = [0] + [x+1 for x in idxs]
            reshaped = reshaped.permute(*permute)

        def prod(values):
            return reduce((lambda x, y: x * y), values)

        reshape_spec = self.reshape_spec
        if self.reshape_spec == 'flatten':
            reshape_spec = [idxs]
        if any(isinstance(x, (list, dict)) for x in reshape_spec):
            reshape = [input.shape[0]]
            for val in reshape_spec:
                if isinstance(val, int):
                    reshape.append(in_dims[val])
                elif isinstance(val, list):
                    reshape.append(prod(in_dims[v] for v in val))
                elif isinstance(val, dict):
                    idx = next(iter(val.keys()))
                    in_dim = in_dims[int(idx)]
                    dims = val[idx]
                    if any(x == auto_tag for x in dims):
                        auto_idx = dims.index(auto_tag)
                        nonauto = prod(x for x in dims if x != auto_tag)
                        dims[auto_idx] = in_dim//nonauto
                    reshape.extend(dims)

            reshaped = reshaped.reshape(reshape)

        return reshaped


class Group(torch.nn.Module):
    """Group module that receives narchi blocks, graph, input and output objects."""

    def __init__(self, block_id, blocks, blocks_mappings, module_cfg, graph, input, output):
        super().__init__()

        from_input = block_id+'¦input'
        from_blocks = [{'_id': from_input}]
        block = {'_id': block_id, 'blocks': blocks, 'graph': graph, 'input': input}
        self.topological_predecessors = parse_graph(from_blocks, block)
        self.input = from_input
        self.output = output
        self.blocks_mappings = blocks_mappings
        self.blocks = get_blocks_dict(blocks)

        for num in range(len(blocks)):
            block_id = id_strip_parent_prefix(blocks[num]._id)
            block = instantiate_block(blocks[num], blocks_mappings, module_cfg)
            setattr(self, block_id, block)


    def forward(self, input):
        """Runs a forward using the group's input and graph."""
        device = next(self.parameters()).device
        values = OrderedDict({self.input: input.to(device)})
        graph_forward(self, values)
        return values[self.output]


def graph_forward(module, values, out_ids=set(), intermediate_outputs=False):
    """Runs a forward for a module using its graph topological_predecessors."""

    if intermediate_outputs and not hasattr(module, 'intermediate_outputs'):
        module.intermediate_outputs = OrderedDict()

    needed_until = OrderedDict()
    for node, inputs in module.topological_predecessors.items():
        for input in inputs:
            needed_until[input] = node

    for node, inputs in module.topological_predecessors.items():
        if node in out_ids:
            values[node] = values[inputs[0]]
            continue

        submodule = getattr(module, id_strip_parent_prefix(node))
        try:
            if isinstance(submodule, BaseModule):
                assert len(inputs) == 1
                assert len(submodule.architecture.inputs) == 1
                result = submodule(**{submodule.architecture.inputs[0]._id: next(iter(values.values()))})
            else:
                result = submodule(*[values[x] for x in inputs])
        except Exception as ex:
            raise type(ex)(f'{type(submodule).__name__}[id={node}]: {ex}') from ex
        #from torchvision.utils import save_image
        #torch.save(result, f'/tmp/new_{node}.pth')
        values[node] = result

        if intermediate_outputs:
            module.intermediate_outputs[node] = result

        for input in inputs:
            if needed_until[input] == node:
                del values[input]


standard_pytorch_blocks_mappings = {
    'Sequential': {
        'class': 'narchi.instantiators.pytorch.Sequential',
    },
    'Add': {
        'class': 'narchi.instantiators.pytorch.Add',
    },
    'Concatenate': {
        'class': 'narchi.instantiators.pytorch.Concatenate',
    },
    'RNN': {
        'class': 'narchi.instantiators.pytorch.RNN',
        'kwargs': {
            'input_size': 'shape:in:1',
            'batch_first': 'const:bool:True',
            ':skip:': 'output_feats',
        },
    },
    'GRU': {
        'class': 'narchi.instantiators.pytorch.GRU',
        'kwargs': {
            'input_size': 'shape:in:1',
            'batch_first': 'const:bool:True',
            ':skip:': 'output_feats',
        },
    },
    'LSTM': {
        'class': 'narchi.instantiators.pytorch.LSTM',
        'kwargs': {
            'input_size': 'shape:in:1',
            'batch_first': 'const:bool:True',
            ':skip:': 'output_feats',
        },
    },
    'Reshape': {
        'class': 'narchi.instantiators.pytorch.Reshape',
    },
    'Group': {
        'class': 'narchi.instantiators.pytorch.Group',
        'kwargs': {
            'block_id': '_id',
        },
    },
    'Module': {
        'class': 'narchi.instantiators.pytorch.StandardModule',
    },
    'Softmax': {
        'class': 'torch.nn.Softmax',
    },
    'LogSoftmax': {
        'class': 'torch.nn.LogSoftmax',
    },
    'ReLU': {
        'class': 'torch.nn.ReLU',
    },
    'LeakyReLU': {
        'class': 'torch.nn.LeakyReLU',
    },
    'Dropout': {
        'class': 'torch.nn.Dropout',
    },
    'Conv1d': {
        'class': 'torch.nn.Conv1d',
        'kwargs': {
            'in_channels': 'shape:in:0',
            'out_channels': 'output_feats',
        },
    },
    'Conv2d': {
        'class': 'torch.nn.Conv2d',
        'kwargs': {
            'in_channels': 'shape:in:0',
            'out_channels': 'output_feats',
        },
    },
    'BatchNorm2d': {
        'class': 'torch.nn.BatchNorm2d',
        'kwargs': {
            'num_features': 'shape:in:0',
        },
    },
    'MaxPool1d': {
        'class': 'torch.nn.MaxPool1d',
    },
    'MaxPool2d': {
        'class': 'torch.nn.MaxPool2d',
        #'kwargs': {
        #    'ceil_mode': 'const:bool:True',
        #},
    },
    'AdaptiveAvgPool1d': {
        'class': 'torch.nn.AdaptiveAvgPool1d',
        'kwargs': {
            'output_size': 'output_feats',
        },
    },
    'AdaptiveAvgPool2d': {
        'class': 'torch.nn.AdaptiveAvgPool2d',
        'kwargs': {
            'output_size': 'output_feats',
        },
    },
    'Embedding': {
        'class': 'torch.nn.Embedding',
        'kwargs': {
            'embedding_dim': 'output_feats',
        },
    },
    'Identity': {
        'class': 'torch.nn.Identity',
    },
    'Linear': {
        'class': 'torch.nn.Linear',
        'kwargs': {
            'in_features': 'shape:in:-1',
            'out_features': 'output_feats',
        },
    },
}


class StandardModule(BaseModule):
    """Class for instantiating pytorch modules with standard block mappings."""
    blocks_mappings = standard_pytorch_blocks_mappings
