"""Functions for symbolic operations."""

import re
import sympy
from sympy.core import numbers
from .schema import variable_pattern


variable_regex = re.compile('^'+variable_pattern+'$')


def sympify_variable(value):
    """Returns the sympyfied object for the given value."""
    var_match = variable_regex.match(str(value))
    if var_match:
        value_sympy = sympy.sympify(var_match[1])
    elif isinstance(value, (int, str)):
        value_sympy = sympy.sympify(value)
    else:
        raise ValueError('Expected input to be an int or a valid sympy expression or a string with '
                         'pattern: '+variable_regex.pattern+', but got '+str(value)+'.')
    return value_sympy


def variable_operate(value, operation):
    """Performs a symbolic operation on a given value.

    Args:
        value (str or int): The value to operate on, either an int or a variable, e.g. "<<variable:W/2+H/4>>".
        operation (str or int): Operation to apply on value, either int or expression, e.g. "__input__/3".

    Returns:
        (str or int): The result of the operation.

    Raises:
        ValueError:
            * If operation is not int nor a valid expression.
            * If value is not an int or a string that follows variable_regex.pattern.
            * If value is not a valid expression or contains "__input__" as a free symbol.
    """
    ## Handle operation ##
    if isinstance(operation, int):
        return operation
    elif not isinstance(operation, str):
        raise ValueError('Expected operation to be an int or a string.')
    operation_sympy = sympify_variable(operation)
    ## Handle input ##
    value_sympy = sympify_variable(value)
    value_symbs = {str(s) for s in value_sympy.free_symbols}
    if '__input__' in value_symbs:
        raise ValueError('Value must not contain "__input__" as a free symbol, but got '+str(input)+'.')
    ## Substitute input into operation ##
    output_sympy = operation_sympy.subs({'__input__': value_sympy})
    ## Return operation result ##
    if isinstance(output_sympy, numbers.Integer):
        return int(output_sympy)
    return '<<variable:'+str(output_sympy).replace(' ', '')+'>>'


def prod(values):
    """Performs a symbolic product of all input values.

    Args:
        values (list[str or int]): List of values to operate on.

    Returns:
        (str or int): The result of the operation.

    Raises:
        ValueError: If any value is not an int or a string that follows variable_regex.pattern.
    """
    if not isinstance(values, list) or len(values) < 1 or not all(isinstance(v, (str, int)) for v in values):
        raise ValueError('Expected values to be a list containing int or str elements, got '+str(values)+'.')
    value = values[0]
    for num in range(1, len(values)):
        value_sympy = sympify_variable(value)
        value = variable_operate(values[num], '__input__*('+str(value_sympy)+')')
    return value
