"""Definition of the narchi json schemas."""

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
dims_in_type = deepcopy(dims_type)
dims_in_type['items']['oneOf'].append({'type': 'null'})


shape_type = {
    'type': 'object',
    'properties': {
        'in':  {'$ref': '#/definitions/dims_in'},
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


reshape_dims_type = deepcopy(dims_type)
reshape_dims_type['minItems'] = 2
reshape_index_type = {
    'type': 'integer',
    'minimum': 0,
}
reshape_flatten_type = {
    'type': 'array',
    'minItems': 2,
    'items': {'$ref': '#/definitions/reshape_index'},
}
reshape_unflatten_type = {
    'type': 'object',
    'minProperties': 1,
    'maxProperties': 1,
    'patternProperties': {
        '^[0-9]+$': {'$ref': '#/definitions/reshape_dims'},
    },
    'additionalProperties': False,
}
reshape_type = {
    'oneOf': [
        {
            'const': 'flatten',
        },
        {
            'type': 'array',
            'minItems': 1,
            'items': {
                'oneOf': [
                    {'$ref': '#/definitions/reshape_index'},
                    {'$ref': '#/definitions/reshape_flatten'},
                    {'$ref': '#/definitions/reshape_unflatten'},
                ],
            },
        },
    ],
}
reshape_definitions = {
    'reshape': reshape_type,
    'reshape_index': reshape_index_type,
    'reshape_dims': reshape_dims_type,
    'reshape_flatten': reshape_flatten_type,
    'reshape_unflatten': reshape_unflatten_type,
}
reshape_schema = {
    '$ref': '#/definitions/reshape',
    'definitions': reshape_definitions,
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
        '_path':        {'$ref': '#/definitions/path'},
        '_ext_vars':    {'type': 'object'},
        'blocks':       {'$ref': '#/definitions/blocks'},
        'input':        {'$ref': '#/definitions/id'},
        'output':       {'$ref': '#/definitions/id'},
        'graph':        {'$ref': '#/definitions/graph'},
        'dim':          {'type': 'integer'},
        'reshape_spec': {'$ref': '#/definitions/reshape'},
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
            'then': {'required': ['graph', 'input', 'output']},
            'else': {'not': {'required': ['graph', 'input', 'output']}},
        },
        {
            'if': {'properties': {'_class': {'const': 'Module'}}},
            'then': {'required': ['_path']},
            'else': {'not': {'required': ['_path', '_ext_vars', 'architecture']}},
        },
        {
            'if': {'properties': {'_class': {'const': 'Sequential'}}},
            'else': {'properties': {'blocks': {'items': {'required': ['_id']}}}},  # not working!
        },
        {
            'if': {'properties': {'_class': {'const': 'Concatenate'}}},
            'then': {'required': ['dim']},
        },
        {
            'if': {'properties': {'_class': {'const': 'Reshape'}}},
            'then': {'required': ['reshape_spec']},
            'else': {'not': {'required': ['reshape_spec']}},
        },
    ],
}


blocks_type = {
    'type': 'array',
    'minItems': 1,
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
    'dims_in': dims_in_type,
    'shape': shape_type,
    'graph': graph_type,
    'block': block_type,
    'blocks': blocks_type,
    'path': path_type,
    'inputs_outputs': inputs_outputs_type,
    'architecture': architecture_type,
}
definitions.update(reshape_definitions)


narchi_schema = {
    '$schema': 'http://json-schema.org/draft-07/schema#',
    '$id': 'https://schema.omnius.com/json/narchi/1.0/schema.json',
    'title': 'Neural Network Module Architecture Schema',
    '$ref': '#/definitions/architecture',
    'definitions': definitions,
}


block_definitions = deepcopy(definitions)
block_definitions['id'] = {'type': 'string', 'pattern': '^'+propagated_id_pattern+'$'}
block_definitions['graph']['items']['pattern'] = '^'+propagated_id_pattern+'( +-> +'+propagated_id_pattern+')+$'


block_schema = {
    '$ref': '#/definitions/block',
    'definitions': block_definitions,
}


propagated_definitions = deepcopy(block_definitions)
propagated_definitions['block']['required'] += ['_shape']
propagated_definitions['dims']['items']['oneOf'][1]['pattern'] = '^'+variable_pattern+'$'
propagated_definitions['architecture']['properties']['_shape'] = {'$ref': '#/definitions/shape'}
propagated_schema = deepcopy(narchi_schema)
propagated_schema['definitions'] = propagated_definitions
propagated_schema['title'] = 'Neural Network Module Propagated Architecture Schema'
del propagated_schema['$id']


mappings_schema = {
    'type': 'object',
    'minProperties': 1,
    'patternProperties': {
        '^'+id_pattern+'$': {
            'type': 'object',
            'properties': {
                'class': {
                    'type': 'string',
                    'pattern': '^[A-Za-z_][0-9A-Za-z_.]*$'
                },
                'kwargs': {
                    'type': 'object',
                    'patternProperties': {
                        '^'+id_pattern+'$': {
                            'type': 'string',
                            'oneOf': [
                                {'pattern': '^'+id_pattern+'$'},
                                {'pattern': '^shape:in:(|-)[0-9]+$'},
                                {'pattern': '^const:str:[^:]+$'},
                                {'pattern': '^const:int:[0-9]+$'},
                                {'pattern': '^const:bool:(True|False)$'},
                            ],
                        },
                        '^:skip:$': {
                            'type': 'string',
                            'pattern': '^'+id_pattern+'$',
                        },
                    },
                    'additionalProperties': False,
                },
                'out_index': {'type': 'integer'},
                'required': ['class'],
                'additionalProperties': False,
            },
        },
    },
    'additionalProperties': False,
}


block_validator = jsonvalidator(block_schema)
narchi_validator = jsonvalidator(narchi_schema)
propagated_validator = jsonvalidator(propagated_schema)
reshape_validator = jsonvalidator(reshape_schema)
mappings_validator = jsonvalidator(mappings_schema)


schemas = {
    None: narchi_schema,
    'narchi': narchi_schema,
    'propagated': propagated_schema,
    'reshape': reshape_schema,
    'block': block_schema,
    'mappings': mappings_schema,
}


def schema_as_str(schema=None):
    """Formats a schema as a pretty printed json string.

    Args:
        schema (str or None): The schema name to return among {'narchi', 'propagated', 'reshape', 'block', 'mappings'}.

    Returns:
        str: Pretty printed schema.
    """
    return json.dumps(schemas[schema], indent=2, ensure_ascii=False)
