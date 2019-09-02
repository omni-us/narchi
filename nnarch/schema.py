"""Definition of the nnarch json schema."""

from jsonargparse import jsonvalidator


description_schema = {
    'type': 'string',
    'minLength': 1,
    'pattern': '.*[^ ].*',
}


id_schema = {
    'type': 'string',
    'pattern': '^[A-Za-z_][0-9A-Za-z_]*$',
}


class_schema = id_schema
class_submodule_schema = dict(class_schema)
class_submodule_schema['not'] = {
    'enum': ['Sequential'],
}


variable_pattern = '<<variable:([-+/*0-9A-Za-z_]+)>>'

dims_schema = {
    'type': 'array',
    'minItems': 1,
    'maxItems': 4,
    'items': {
        'oneOf': [
            {
                'type': 'integer',
                'minimum': 1,
            },
            {
                'type': 'string',
                'pattern': '^('+variable_pattern+'|<<auto>>)$',
            },
        ],
    },
}


shape_schema = {
    'oneOf': [
        dims_schema,
        {
            'type': 'object',
            'required': ['in', 'out'],
            'additionalProperties': False,
            'properties': {
                'in': dims_schema,
                'out': dims_schema,
            },
        },
    ],
}


submodule_schema = {
    'type': 'object',
    'required': ['_class'],
    'properties': {
        '_class': class_submodule_schema,
        '_description': description_schema,
        '_id': id_schema,
        '_shape': shape_schema,
    },
}


block_schema = {
    'type': 'object',
    'required': ['_class', '_id'],
    'properties': {
        '_class': class_schema,
        '_description': description_schema,
        '_id': id_schema,
        '_shape': shape_schema,
        'blocks': {
            'type': 'array',
            'minItems': 1,
            'items': submodule_schema,
        },
    },
    #'if': {
    #    'properties': {
    #        '_class': {
    #            'enum': ['Sequential'],
    #        },
    #    },
    #},
    #'then': {
    #    'required': ['blocks'],
    #    'properties': {
    #        'blocks': {
    #            'type': 'array',
    #            'minItems': 1,
    #            'items': submodule_schema,
    #        },
    #    },
    #},
}


input_output_schema = {
    'type': 'array',
    'minItems': 1,
    'maxItems': 1,
    'items': {
        'type': 'object',
        'required': ['_id', '_shape'],
        'additionalProperties': False,
        'properties': {
            '_description': description_schema,
            '_id': id_schema,
            '_shape': dims_schema,
        },
    },
}


nnarch_schema = {
    'type': 'object',
    'required': ['blocks', 'graph', 'inputs', 'outputs'],
    'additionalProperties': False,
    'properties': {
        '_description': description_schema,
        'blocks': {
            'type': 'array',
            'minItems': 1,
            'items': block_schema,
        },
        'graph': {
            'type': 'array',
            'minItems': 1,
            'items': {'type': 'string'},
        },
        'inputs': input_output_schema,
        'outputs': input_output_schema,
    },
}


nnarch_validator = jsonvalidator(nnarch_schema)
