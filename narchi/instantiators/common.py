"""Generic code for module architecture instantiators."""

import inspect
from ..schemas import mappings_validator, id_separator
from ..propagators.base import get_shape


def id_strip_parent_prefix(value):
    """Removes the parent prefix from an id value."""
    return value.rsplit(id_separator)[-1]


def import_object(name):
    """Function that returns a class in a module given its dot import statement."""
    name_module, name_class = name.rsplit('.', 1)
    module = __import__(name_module, fromlist=[name_class])
    return getattr(module, name_class)


def instantiate_block(block_cfg, blocks_mappings, module_cfg):
    """Function that instantiates a block given its narchi config and a mappings object."""
    mappings_validator.validate(blocks_mappings)
    if block_cfg._class not in blocks_mappings:
        raise NotImplementedError('No mapping for blocks of type '+block_cfg._class+'.')

    kwargs = {k: v for k, v in vars(block_cfg).items() if not k.startswith('_')}

    def set_kwargs(key_to, key_from, value=None):
        if key_to in kwargs:
            print('warning: mapping defines '+key_to+' as '+key_from+' so replacing current value: '+str(kwargs[key_to])+'.')
        if value is None:
            kwargs[key_to] = kwargs.pop(key_from)
        else:
            kwargs[key_to] = value

    block_mapping = blocks_mappings[block_cfg._class]
    block_class = import_object(block_mapping['class'])
    if 'kwargs' in block_mapping:
        for key_to, key_from in block_mapping['kwargs'].items():
            if key_to == ':skip:':
                del kwargs[key_from]
            elif key_from[0] == '_':
                set_kwargs(key_to, key_from, getattr(block_cfg, key_from))
            elif key_from.startswith('shape:'):
                _, key, idx = key_from.split(':')
                set_kwargs(key_to, key_from, get_shape(key, block_cfg)[int(idx)])
            elif key_from.startswith('const:'):
                _, vtype, val = key_from.split(':')
                if vtype == 'int':
                    val = int(val)
                elif vtype == 'bool':
                    val = False if val == 'False' else True
                set_kwargs(key_to, key_from, val)
            else:
                set_kwargs(key_to, key_from)

    if block_cfg._class == 'Module':
        set_kwargs('cfg', 'module_cfg', module_cfg)

    func_param = {x.name for x in inspect.signature(block_class).parameters.values()}
    if 'blocks_mappings' in func_param:
        set_kwargs('blocks_mappings', 'function_parameter', blocks_mappings)
    if 'module_cfg' in func_param:
        set_kwargs('module_cfg', 'function_parameter', module_cfg)

    return block_class(**kwargs)
