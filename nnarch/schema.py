"""Definition of the nnarch json schema."""

import json
from jsonschema import Draft7Validator as jsonvalidator


id_pattern = '[A-Za-z_][0-9A-Za-z_]*'
variable_pattern = '<<variable:([-+/*0-9A-Za-z_]+)>>'


id_type = {
    'type': 'string',
    'pattern': '^'+id_pattern+'$',
}


description_type = {
    'type': 'string',
    'minLength': 8,
    'pattern': '.*[^ ].*',
}


dims_type = {
    'type': 'array',
    'minItems': 1,
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


shape_type = {
    'type': 'object',
    'properties': {
        'in':  {'$ref': '#/definitions/dims'},
        'out': {'$ref': '#/definitions/dims'},
    },
    'required': ['in', 'out'],
    'additionalProperties': False,
}


graph_type = {
    'type': 'array',
    'minItems': 1,
    'items': {
        'type': 'string',
        'pattern': '^'+id_pattern+'( +-> +'+id_pattern+')+$',
    },
}


path_type = {
    'type': 'string',
    'pattern': '.+\\.jsonnet',
}


block_type = {
    'type': 'object',
    'properties': {
        '_class':        {'$ref': '#/definitions/id'},
        '_name':         {'$ref': '#/definitions/id'},
        '_id':           {'$ref': '#/definitions/id'},
        '_id_is_global': {'type': 'boolean'},
        '_description':  {'$ref': '#/definitions/description'},
        '_shape':        {'$ref': '#/definitions/shape'},
        'path':          {'$ref': '#/definitions/path'},
        'ext_vars':      {'type': 'object'},
        'blocks':        {'$ref': '#/definitions/blocks'},
        'graph':         {'$ref': '#/definitions/graph'},
    },
    'required': ['_class', '_id'],
    'allOf': [
        {
            'if': {'properties': {'_class': {'enum': ['Sequential', 'Group']}}},
            'then': {'required': ['blocks']},
            'else': {'not': {'required': ['blocks']}},
        },
        {
            'if': {'properties': {'_class': {'const': 'Module'}}},
            'then': {'required': ['path']},
            'else': {'not': {'required': ['path', 'ext_vars']}},
        },
        {
            'if': {'properties': {'_class': {'const': 'Group'}}},
            'then': {'required': ['graph']},
            'else': {'not': {'required': ['graph']}},
        },
    ],
}


blocks_type = {
    'type': 'array',
    'minItems': 2,
    'items': {'$ref': '#/definitions/block'},
}


inputs_outputs_type = {
    'type': 'array',
    'minItems': 1,
    'items': {
        'type': 'object',
        'properties': {
            '_id':          {'$ref': '#/definitions/id'},
            '_description': {'$ref': '#/definitions/description'},
            '_shape':       {'$ref': '#/definitions/dims'},
        },
        'required': ['_id', '_shape'],
        'additionalProperties': False,
    },
}


block_definitions = {
    'id': id_type,
    'description': description_type,
    'dims': dims_type,
    'shape': shape_type,
    'graph': graph_type,
    'block': block_type,
    'blocks': blocks_type,
    'path': path_type,
}


nnarch_definitions = dict(block_definitions)
nnarch_definitions.update({
    'inputs_outputs': inputs_outputs_type,
})


block_schema = {
    '$ref': '#/definitions/block',
    'definitions': block_definitions,
}


nnarch_schema = {
    '$schema': 'http://json-schema.org/draft-07/schema#',
    '$id': 'https://schema.omnius.com/json/nnarch/0.0/schema.json',
    'title': 'omni:us Neural Network Module Architecture Schema',
    'type': 'object',
    'properties': {
        '_id':          {'$ref': '#/definitions/id'},
        '_description': {'$ref': '#/definitions/description'},
        'blocks':       {'$ref': '#/definitions/blocks'},
        'graph':        {'$ref': '#/definitions/graph'},
        'inputs':       {'$ref': '#/definitions/inputs_outputs'},
        'outputs':      {'$ref': '#/definitions/inputs_outputs'},
    },
    'required': ['_id', 'blocks', 'graph', 'inputs', 'outputs'],
    'additionalProperties': False,
    'definitions': nnarch_definitions,
}


block_validator = jsonvalidator(block_schema)
nnarch_validator = jsonvalidator(nnarch_schema)


def schema_as_str(schema=nnarch_schema):
    """Returns the schema as a pretty printed json string."""
    return json.dumps(schema, indent=2)
