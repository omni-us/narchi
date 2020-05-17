"""A standard module instantiator for pytorch."""

import torch
from functools import reduce
from collections import OrderedDict
from jsonargparse import Path, config_read_mode
from .common import instantiate_block, id_strip_parent_prefix
from ..module import ModuleArchitecture
from ..propagators.reshape import check_reshape_spec, norm_reshape_spec
from ..propagators.group import get_blocks_dict
from ..graph import parse_graph


class BaseModule(torch.nn.Module, ModuleArchitecture):
    """Base class for pytorch modules based on an narchi architecture."""

    def __init__(self, *args, state_dict=None, **kwargs):
        if 'cfg' not in kwargs:
            kwargs['cfg'] = {'propagators': 'default'}
        elif 'propagators' not in kwargs['cfg']:
            kwargs['cfg']['propagators'] = 'default'
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


    def forward(self, *args, **kwargs):
        """Runs a forward using the architecture's inputs and graph."""
        inputs = {x for x in kwargs.keys()}
        expected_inputs = {x._id for x in self.architecture.inputs}
        if len(args) != 0:
            raise RuntimeError(type(self).__name__+' expects only keyword arguments.')
        if inputs != expected_inputs:
            raise RuntimeError(type(self).__name__+' got unexpected arguments, given '+str(inputs)+', expected '+str(expected_inputs)+'.')

        device = next(self.parameters()).device
        values = OrderedDict({x: kwargs[x].to(device) for x in inputs})

        out_ids = {x._id for x in self.architecture.outputs}
        graph_forward(self, values, out_ids)

        if len(self.architecture.outputs) == 1:
            return values[self.architecture.outputs[0]._id]
        else:
            return [values[x._id] for x in self.architecture.outputs]


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
        elif isinstance(state_dict, (str, Path)):
            path = Path(state_dict, mode=config_read_mode, cwd=self.cfg.cwd)
            state_dict = torch.load(path())
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


class Reshape(torch.nn.Module):
    """Reshape module that receives as input an narchi reshape_spec object."""

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
            raise RuntimeError(type(self).__name__+' got a tensor with '+str(len(input.shape))+' dimensions but expected '+str(len(idxs)+1)+'.')

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
                    if any(x == '<<auto>>' for x in dims):
                        auto_idx = dims.index('<<auto>>')
                        nonauto = prod(x for x in dims if x != '<<auto>>')
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


def graph_forward(module, values, out_ids=set()):
    """Runs a forward for a module using its graph topological_predecessors."""
    for node, inputs in module.topological_predecessors.items():
        if node in out_ids:
            values[node] = values[inputs[0]]
            continue
        submodule = getattr(module, id_strip_parent_prefix(node))
        if isinstance(submodule, BaseModule):
            assert len(inputs) == 1
            assert len(submodule.architecture.inputs) == 1
            result = submodule(**{submodule.architecture.inputs[0]._id: next(iter(values.values()))})
        else:
            result = submodule(*[values[x] for x in inputs])
        block_mapping = module.blocks_mappings[module.blocks[node]._class]
        if 'out_index' in block_mapping:
            values[node] = result[block_mapping['out_index']]
        else:
            values[node] = result


standard_pytorch_blocks_mappings = {
    'Sequential': {
        'class': 'narchi.instantiators.pytorch.Sequential',
    },
    'Add': {
        'class': 'narchi.instantiators.pytorch.Add',
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
        'kwargs': {
            'output_size': 'output_feats',
        },
    },
    'LSTM': {
        'class': 'torch.nn.LSTM',
        'kwargs': {
            'input_size': 'shape:in:1',
            'batch_first': 'const:bool:True',
            ':skip:': 'output_feats',
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
            'out_features': 'output_feats',
        },
    },
}


class StandardModule(BaseModule):
    blocks_mappings = standard_pytorch_blocks_mappings
