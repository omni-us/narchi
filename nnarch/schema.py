"""Definition of the nnarch json schema."""

import json
from jsonschema import Draft7Validator as jsonvalidator


id_pattern = '[A-Za-z_][0-9A-Za-z_]*'
variable_pattern = '<<variable:([-+/*0-9A-Za-z_]+)>>'


description_schema = {
    'type': 'string',
    'minLength': 1,
    'pattern': '.*[^ ].*',
}


id_schema = {
    'type': 'string',
    'pattern': '^'+id_pattern+'$',
}


class_schema = id_schema
class_submodule_schema = dict(class_schema)
class_submodule_schema['not'] = {
    'enum': ['Sequential'],
}


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


blocks_schema = {
    'type': 'array',
    'minItems': 1,
    'items': block_schema,
}


graph_schema = {
    'type': 'array',
    'minItems': 1,
    'items': {
        'type': 'string',
        'pattern': '^'+id_pattern+'( -> '+id_pattern+')+$',
    },
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
    '$schema': 'http://json-schema.org/draft-07/schema#',
    '$id': 'https://schema.omnius.com/json/nnarch/1.0/schema.json',
    'title': 'omni:us Neural Network Module Architecture Schema',
    'type': 'object',
    'required': ['blocks', 'graph', 'inputs', 'outputs'],
    'additionalProperties': False,
    'properties': {
        '_description': description_schema,
        'blocks': blocks_schema,
        'graph': graph_schema,
        'inputs': input_output_schema,
        'outputs': input_output_schema,
    },
}


nnarch_validator = jsonvalidator(nnarch_schema)


def schema_as_str():
    """Returns the nnarch schema as a pretty printed json string."""
    return json.dumps(nnarch_schema, indent=2, sort_keys=True)
