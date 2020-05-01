"""Definition of the nnarch json schema."""

import json
from copy import deepcopy
from jsonschema import Draft7Validator as jsonvalidator


id_pattern = '[A-Za-z_][0-9A-Za-z_]*'
id_separator = '·'
propagated_id_pattern = '[A-Za-z_][0-9A-Za-z_'+id_separator+']*'
variable_pattern = '<<variable:([-+/*0-9A-Za-z_]+)>>'


id_type = {
    'type': 'string',
    'pattern': '^'+id_pattern+'$',
}


description_type = {
    'type': 'string',
    'minLength': 8,
    'pattern': '^[^<>]+$',
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
        '_class':       {'$ref': '#/definitions/id'},
        '_name':        {'$ref': '#/definitions/id'},
        '_id':          {'$ref': '#/definitions/id'},
        '_id_share':    {'$ref': '#/definitions/id'},
        '_description': {'$ref': '#/definitions/description'},
        '_shape':       {'$ref': '#/definitions/shape'},
        'blocks':       {'$ref': '#/definitions/blocks'},
        'graph':        {'$ref': '#/definitions/graph'},
        'path':         {'$ref': '#/definitions/path'},
        'ext_vars':     {'type': 'object'},
        'architecture': {'$ref': '#/definitions/architecture'},
    },
    #'required': ['_class', '_id'],
    'required': ['_class'],
    'allOf': [
        {
            'if': {'properties': {'_class': {'enum': ['Sequential', 'Group']}}},
            'then': {'required': ['blocks']},
            'else': {'not': {'required': ['blocks']}},
        },
        {
            'if': {'properties': {'_class': {'const': 'Group'}}},
            'then': {'required': ['graph']},
            'else': {'not': {'required': ['graph']}},
        },
        {
            'if': {'properties': {'_class': {'const': 'Module'}}},
            'then': {'required': ['path']},
            'else': {'not': {'required': ['path', 'ext_vars', 'architecture']}},
        },
        {
            'if': {'properties': {'_class': {'const': 'Sequential'}}},
            'else': {'properties': {'blocks': {'items': {'required': ['_id']}}}},  # not working!
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


architecture_type = {
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
}


definitions = {
    'id': id_type,
    'description': description_type,
    'dims': dims_type,
    'shape': shape_type,
    'graph': graph_type,
    'block': block_type,
    'blocks': blocks_type,
    'path': path_type,
    'inputs_outputs': inputs_outputs_type,
    'architecture': architecture_type,
}


block_schema = {
    '$ref': '#/definitions/block',
    'definitions': definitions,
}


nnarch_schema = {
    '$schema': 'http://json-schema.org/draft-07/schema#',
    '$id': 'https://schema.omnius.com/json/nnarch/0.0/schema.json',
    'title': 'Neural Network Module Architecture Schema',
    '$ref': '#/definitions/architecture',
    'definitions': definitions,
}


propagated_definitions = deepcopy(definitions)
propagated_definitions['id'] = {'type': 'string', 'pattern': '^'+propagated_id_pattern+'$'}
propagated_definitions['graph']['items']['pattern'] = '^'+propagated_id_pattern+'( +-> +'+propagated_id_pattern+')+$'
propagated_definitions['block']['required'] += ['_shape']
propagated_definitions['dims']['items']['oneOf'][1]['pattern'] = '^'+variable_pattern+'$'
propagated_schema = deepcopy(nnarch_schema)
propagated_schema['definitions'] = propagated_definitions
propagated_schema['title'] = 'Neural Network Module Propagated Architecture Schema'
del propagated_schema['$id']


block_validator = jsonvalidator(block_schema)
nnarch_validator = jsonvalidator(nnarch_schema)
propagated_validator = jsonvalidator(propagated_schema)


def schema_as_str(schema=None):
    """Returns the schema as a pretty printed json string."""
    if schema is None:
        schema = nnarch_schema
    return json.dumps(schema, indent=2)
