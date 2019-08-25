
import re
from collections import OrderedDict
from yamlargparse import SimpleNamespace, ActionJsonnet, jsonvalidator, namespace_to_dict
from pygraphviz import AGraph
from networkx.drawing.nx_agraph import from_agraph
from networkx.algorithms.dag import is_directed_acyclic_graph
from networkx.algorithms.traversal.edgebfs import edge_bfs
from copy import deepcopy
from .schema import nnarch_schema


nnarch_validator = jsonvalidator(nnarch_schema)


def parse_architecture(architecture, const={}):
    """Parses a neural network architecture.

    Args:
        architecture (SimpleNamespace or str): A parsed architecture namespace object or path to a jsonnet architecture file.
        const (dict): Dictionary of constant values for replacement in architecture.

    Returns:
        (SimpleNamespace, OrderedDict, OrderedDict): A tuple with elements:
            1) The completed architecture object.
            2) An ordered (as defined in architecture) dictionary of the network blocks including inputs and outputs.
            3) An ordered (by graph traversal) dictionary of network block ids mapping to its inputs.

    Raises:
        NotImplementedError: If the parsed architecture is not supported.
        ValueError: If there is a problem with the architecture.
    """
    ## Load file or snippet or make copy of object ##
    if isinstance(architecture, str):
        architecture = ActionJsonnet(schema=None).parse(architecture)
    else:
        architecture = deepcopy(architecture)

    ## Validate input ##
    try:
        nnarch_validator.validate(namespace_to_dict(architecture))
    except Exception as ex:
        raise type(ex)('Input architecture does not validate against nnarch schema :: '+str(ex))

    ## Check if supported ##
    if len(architecture.inputs) != 1:
        raise NotImplementedError('Architectures with more than one input not implemented.')
    if len(architecture.outputs) != 1:
        raise NotImplementedError('Architectures with more than one output not implemented.')

    ## Parse graph ##
    try:
        graph = from_agraph(AGraph('\n'.join(['digraph {']+architecture.graph+['}'])))
    except Exception as ex:
        raise ValueError('Problems parsing architecture graph: '+str(ex))
    if not is_directed_acyclic_graph(graph):
        raise ValueError('Expected architecture graph to be directed and acyclic.')

    ## Traverse graph ##
    try:
        in_nodes = _get_nodes_with_inputs(graph, source=architecture.inputs[0]._id)
    except Exception as ex:
        raise ValueError('Problems traversing architecture graph: '+str(ex))
    if len(in_nodes) != graph.number_of_nodes()-len(architecture.inputs):
        raise ValueError('Graph traversal does not include all nodes: '+str(in_nodes))
    if next(reversed(in_nodes)) != architecture.outputs[0]._id:
        raise ValueError('Expected output node to be the last in the graph.')

    ## Create dictionary of blocks ##
    blocks = OrderedDict()
    input_node = architecture.inputs[0]._id
    _set_shape_const(architecture.inputs[0], const)
    blocks[input_node] = architecture.inputs[0]
    for block in architecture.blocks:
        bid = block._id
        blocks[bid] = block
    output_node = architecture.outputs[0]._id
    _set_shape_const(architecture.outputs[0], const)
    blocks[output_node] = architecture.outputs[0]

    ## Propagate output features to pre-output block ##
    pre_output_block = blocks[in_nodes[output_node][0]]
    output_block = blocks[output_node]
    if (not hasattr(pre_output_block, '_shape') or pre_output_block._shape == '<<auto>>') and not hasattr(pre_output_block, 'out_features'):
        pre_output_block.out_features = output_block._shape[-1]

    ## Complete the architecture blocks ##
    for node_to, nodes_from in in_nodes.items():
        if node_to != architecture.outputs[0]._id:
            blocks[node_to] = _complete_block([blocks[n] for n in nodes_from], blocks[node_to], const)

    ## Automatic output dimensions ##
    for dim, val in enumerate(output_block._shape):
        if val == '<<auto>>':
            output_block._shape[dim] = get_shape('out', pre_output_block)[dim]

    ## Check that output shape agrees ##
    if not _shapes_agree(pre_output_block, output_block):
        raise ValueError('Output shape does not agree: '+str(pre_output_block._shape['out'])+' vs. '+str(output_block._shape))

    ## Validate result ##
    try:
        nnarch_validator.validate(namespace_to_dict(architecture))
    except Exception as ex:
        raise type(ex)('Completed architecture does not validate against nnarch schema :: '+str(ex))

    return architecture, blocks, in_nodes


def _get_nodes_with_inputs(graph, source):
    """Traverses a graph creating an OrderedDict of nodes with respective inputs."""
    in_nodes = OrderedDict()
    for node_from, node_to, _ in edge_bfs(graph, source=source):
        if node_to not in in_nodes:
            in_nodes[node_to] = []
        in_nodes[node_to].append(node_from)
    return in_nodes


def get_shape(key, shape):
    """Gets the shape list for a given key={'in','out'}."""
    if isinstance(shape, SimpleNamespace) and hasattr(shape, '_shape'):
        shape = shape._shape
    if isinstance(shape, dict):
        return shape[key]
    return shape


def _shapes_agree(shape_from, shape_to):
    """Checks whether the output shape from a block agrees with input shape of another block."""
    return get_shape('out', shape_from) == get_shape('in', shape_to)


def _create_shape(shape_in, shape_out=None):
    """Creates a shape dictionary with 'in' and 'out' keys and copied dimensions arrays."""
    return deepcopy({'in': shape_in, 'out': deepcopy(shape_in) if shape_out is None else shape_out})


def _set_shape_dim(key, shape, dim, val, fact=None):
    """Sets a value for a given dimension, shape and key ('in' or 'out')."""
    if isinstance(shape, SimpleNamespace) and hasattr(shape, '_shape'):
        shape = shape._shape
    if fact is not None:
        assert fact[0] in {'/', '*'}, 'Expected factor to start with "/" or "*" but got '+fact
        fact_val = int(fact[1:])
        if isinstance(val, int):
            if fact[0] == '/' and val % fact_val != 0:
                raise ValueError('Dimension '+str(val)+' not divisible by factor '+str(fact_val))
            val = val//fact_val if fact[0] == '/' else val*fact
        elif str(val).startswith('<<variable'):
            if val == '<<variable>>':
                val = '<<variable'+fact+'>>'
            else:
                cur_fact = val.replace('<<variable', '').replace('>>', '')
                cur_fact_val = int(cur_fact[1:])
                if cur_fact[0] == fact[0]:
                    val = '<<variable'+fact[0]+str(cur_fact_val*fact_val)+'>>'
                else:
                    raise NotImplementedError('Combining factors of different types not implemented: '+cur_fact+' + '+fact+'.')
        else:
            raise ValueError('Unexpected value ('+str(val)+') to apply factor '+fact)
    if isinstance(shape, dict):
        shape[key][dim] = val
    else:
        shape[dim] = val


def _set_shape_const(shape, const):
    """Replaces <<const:*>> values in shape objects or individual strings.

    Args:
        shape (SimpleNamespace or dict or str or int): Object on which to do replacement.
        const (dict): Dictionary of constant values for replacement in architecture.

    Returns:
        None or in case replacing individual string, the replaced string or unchanged int.
    """
    if isinstance(shape, SimpleNamespace) and not hasattr(shape, '_shape'):
        return
    if isinstance(shape, int):
        return shape
    if isinstance(shape, str):
        const_match = re.match('^<<const:(\w+)>>$', shape)
        if const_match:
            key = const_match.groups()[0]
            if key not in const:
                raise KeyError('Unprovided constant: '+shape+'.')
            shape = const[key]
        return shape
    if isinstance(shape, SimpleNamespace) and hasattr(shape, '_shape'):
        shape = shape._shape
    for dims in ([shape['in'], shape['out']] if isinstance(shape, dict) else [shape]):
        for dim in range(len(dims)):
            dims[dim] = _set_shape_const(dims[dim], const)


def _complete_block(from_block, block, const):
    """Completes a block inferring input and output shapes and replacing constants.

    Args:
        from_block (SimpleNamespace): Block object that connects with the one being completed.
        block (SimpleNamespace): Block being completed.
        const (dict): Dictionary of constant values for replacement in architecture.
    """
    if isinstance(from_block, list):
        if len(from_block) > 1:
            raise NotImplementedError('More than one input blocks not implemented.')
        from_block = from_block[0]
    from_shape = get_shape('out', from_block)
    if from_shape == '<<auto>>' or (isinstance(from_shape, list) and any([x == '<<auto>>' for x in from_shape])):
        raise ValueError('Input block is not allowed to have <<auto>> values in shape, found for '+from_block._id+' -> '+block._id+'.')
    if any([str(x).startswith('<<const:') for x in from_shape]):
        raise ValueError('Input block is not allowed to have <<const:*>> values in shape, found for '+from_block._id+' -> '+block._id+'.')
    if hasattr(block, '_shape') and isinstance(block._shape, SimpleNamespace):
        block._shape = vars(block._shape)
    _set_shape_const(block, const)

    if block._class == 'Sequential':
        block.modules[0] = _complete_block(from_block, block.modules[0], const)
        for num in range(1, len(block.modules)):
            block.modules[num] = _complete_block(block.modules[num-1], block.modules[num], const)
        if hasattr(block, '_shape'):
            if not _shapes_agree(from_shape, block):
                raise ValueError('Shapes do not agree between output of '+from_block._id+' ('+str(from_shape)+') and input of '+block._id+' ('+str(get_shape('in', block))+').')
        else:
            block._shape = _create_shape(from_shape, get_shape('out', block.modules[-1]))

    elif block._class in {'Conv2d', 'BatchNorm2d', 'MaxPool2d'}:
        if len(from_shape) != 3:
            raise ValueError(block._class+' requires input to have 3 dimensions.')
        if not hasattr(block, '_shape') or block._shape == '<<auto>>':
            if block._class == 'MaxPool2d':
                block._shape = _create_shape(from_shape, [from_shape[0], '<<auto>>', '<<auto>>'])
            elif block._class == 'Conv2d':
                if not hasattr(block, 'out_features'):
                    raise ValueError(block._class+' requires out_features or _shape to be defined.')
                block.out_features = _set_shape_const(block.out_features, const)
                block._shape = _create_shape(from_shape, [block.out_features, '<<auto>>', '<<auto>>'])
            else:
                block._shape = _create_shape(from_shape)
        for dim, val in enumerate(get_shape('in', block)):
            if val == '<<auto>>':
                _set_shape_dim('in', block, dim, from_shape[dim])
        if any([x == '<<auto>>' for x in get_shape('out', block)[1:]]):
            if block._class == 'Conv2d':
                if block.kernel_size//2 != block.padding:
                    raise ValueError('<<auto>> output dims only implemented for size preserving '+block._class+'.')
                for dim, val in enumerate(get_shape('out', block)):
                    if dim != 0 and val == '<<auto>>':
                        _set_shape_dim('out', block, dim, get_shape('in', block)[dim])
            elif block._class == 'MaxPool2d':
                if block.kernel_size != block.stride:
                    raise ValueError('<<auto>> output dims only implemented for kernel_size==block.stride '+block._class+'s.')
                for dim, val in enumerate(get_shape('out', block)):
                    if val == '<<auto>>':
                        in_dim = get_shape('in', block)[dim]
                        #if in_dim != '<<variable>>':
                        #    _set_shape_dim('out', block, dim, in_dim//(1 if dim==0 else block.kernel_size))
                        #elif in_dim.startswith('<<variable'):
                        _set_shape_dim('out', block, dim, in_dim, '/'+str(block.kernel_size))
            else:
                raise ValueError('<<auto>> output dims not implemented for '+block._class+'.')
        #for dim, val in enumerate(get_shape('out', block)):
        #    if val == '<<auto>>':
        #        _set_shape_dim('out', block, dim, from_shape[dim])
        if block._class in {'Conv2d', 'BatchNorm2d'} and not hasattr(block, 'out_features'):
            block.out_features = get_shape('out', block)[0]

    elif block._class in {'LeakyReLU'}:
        if not hasattr(block, '_shape') or block._shape == '<<auto>>':
            block._shape = _create_shape(from_shape)
        else:
            raise NotImplementedError(block._class+' with custom shape is not implemented.')

    elif block._class in {'Reshape2dTo1d'}:
        if len(from_shape) != 3:
            raise ValueError(block._class+' requires input to have 3 dimensions.')
        if not hasattr(block, 'dims'):
            block.dims = [2, [0, 1]]
        if len(block.dims) != 2 or len(block.dims[0] if isinstance(block.dims[0], list) else block.dims[1]) != 2:
            raise ValueError(block._class+' requires dims to be a list with two elements, one an integer and the other a list with two integers.')
        # @todo Generalize for different dims
        if hasattr(block, '_shape'):
            raise ValueError('_shape not allowed for '+block._class+'.')
        if not (isinstance(from_shape[0], int) and isinstance(from_shape[1], int)):
            raise ValueError(block._class+' requires first two input dimension to be fixed, but got '+str(from_shape)+'.')
        block._shape = _create_shape(from_shape, [from_shape[2], from_shape[0]*from_shape[1]])

    elif block._class in {'LSTM'}:
        if len(from_shape) != 2:
            raise ValueError(block._class+' requires input to have 2 dimensions.')
        if not hasattr(block, '_shape') or block._shape == '<<auto>>':
            if not hasattr(block, 'out_features'):
                raise ValueError(block._class+' requires out_features or _shape to be defined.')
            block.out_features = _set_shape_const(block.out_features, const)
            block._shape = _create_shape(from_shape, ['<<auto>>', block.out_features])
        if 'in' not in block._shape or 'out' not in block._shape or len(block._shape['in']) != 2 or len(block._shape['out']) != 2:
            raise ValueError(block._class+' requires _shape to have independent in and out definitions and both have 2 dimensions.')
        if not isinstance(block._shape['out'][1], int):
            raise ValueError('Second output dimension of '+block._class+' should be constant, but got '+str(block._shape)+'.')
        if block.bidirectional and block._shape['out'][1] % 2 != 0:
            raise ValueError('For bidirectional '+block._class+' the second output dimension should be multiple of 2, but got '+str(block._shape['out'][1])+'.')
        for dim, val in enumerate(get_shape('in', block)):
            if val == '<<auto>>':
                _set_shape_dim('in', block, dim, from_shape[dim])
        for dim, val in enumerate(get_shape('out', block)):
            if val == '<<auto>>':
                _set_shape_dim('out', block, dim, from_shape[dim])
        block.hidden_size = block._shape['out'][1] // (2 if block.bidirectional else 1)

    elif block._class in {'Linear'}:
        if not hasattr(block, '_shape') or block._shape == '<<auto>>':
            if not hasattr(block, 'out_features'):
                raise ValueError(block._class+' requires out_features or _shape to be defined.')
            block.out_features = _set_shape_const(block.out_features, const)
            block._shape = _create_shape(from_shape, ['<<auto>>' for n in range(len(from_shape)-1)] + [block.out_features])
        if len(from_shape) != len(get_shape('in', block)):
            raise ValueError('Number of input dimensions does not agree for '+block._class+' '+block._id+'.')
        if len(get_shape('in', block)) != len(get_shape('in', block)):
            raise ValueError('Number of input and output dimensions does not agree for '+block._class+' '+block._id+'.')
        if get_shape('in', block)[-1] == '<<auto>>':
            get_shape('in', block)[-1] = from_shape[-1]
        for dim, val in enumerate(get_shape('out', block)):
            if val == '<<auto>>':
                _set_shape_dim('out', block, dim, from_shape[dim])
        block.in_features = get_shape('in', block)[-1]
        block.out_features = get_shape('out', block)[-1]

    else:
        raise NotImplementedError('Completion of '+block._class+' not implemented.')

    if not _shapes_agree(from_shape, block):
        raise ValueError('Shapes do not agree for block shape_in='+str(from_shape)+' block='+str(block))

    return block
