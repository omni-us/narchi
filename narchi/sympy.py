"""Functions for symbolic operations."""

import re
import sympy
from sympy.core import numbers
from .schemas import variable_pattern


variable_regex = re.compile('^'+variable_pattern+'$')


def is_valid_dim(value):
    """Checks whether value is an int > 0 or str that follows variable_regex.pattern."""
    if isinstance(value, int) or variable_regex.match(str(value)):
        if isinstance(value, int) and value <= 0:
            return False
        return True
    return False


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


def get_nonrational_variable(value):
    """Returns either an int or a string variable."""
    if isinstance(value, numbers.Integer):
        return int(value)
    elif isinstance(value, numbers.Rational):
        raise ValueError('Obtained a rational result: '+str(value)+'.')
    return '<<variable:'+str(value).replace(' ', '')+'>>'


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
    return get_nonrational_variable(output_sympy)


def variables_aggregate(values, operation):
    """Performs a symbolic aggregation operation over all input values.

    Args:
        values (list[str or int]): List of values to operate on.
        operation (str): One of '+'=sum, '*'=prod.

    Returns:
        (str or int): The result of the operation.

    Raises:
        ValueError: If any value is not an int or a string that follows variable_regex.pattern.
    """
    operations = {'+', '*'}
    if operation not in operations:
        raise ValueError('Expected operation to be one of '+str(operations)+', got '+str(operation)+'.')
    if not isinstance(values, list) or len(values) < 1 or not all(isinstance(v, (str, int)) for v in values):
        raise ValueError('Expected values to be a list containing int or str elements, got '+str(values)+'.')
    value = values[0]
    for num in range(1, len(values)):
        value_sympy = sympify_variable(value)
        value = variable_operate(values[num], '__input__'+operation+'('+str(value_sympy)+')')
    return value


def sum(values):
    """Performs a symbolic sum of all input values.

    Args:
        values (list[str or int]): List of values to operate on.

    Returns:
        (str or int): The result of the operation.

    Raises:
        ValueError: If any value is not an int or a string that follows variable_regex.pattern.
    """
    return variables_aggregate(values, '+')


def prod(values):
    """Performs a symbolic product of all input values.

    Args:
        values (list[str or int]): List of values to operate on.

    Returns:
        (str or int): The result of the operation.

    Raises:
        ValueError: If any value is not an int or a string that follows variable_regex.pattern.
    """
    return variables_aggregate(values, '*')


def divide(numerator, denominator):
    """Performs a symbolic division.

    Args:
        numerator (str or int): Value for numerator.
        denominator (str or int): Value for denominator.

    Returns:
        (str or int): The result of the operation.

    Raises:
        ValueError: If any value is not an int or a string that follows variable_regex.pattern.
    """
    numerator = sympify_variable(numerator)
    denominator = sympify_variable(denominator)
    result = numerator/denominator
    return get_nonrational_variable(result)


def conv_out_length(length, kernel, stride, padding, dilation):
    """Performs a symbolic calculation of the output length of a convolution.

    Args:
        length (str or int): Length of the input, either an int or a variable.
        kernel (int): Size of the kernel in the direction of length.
        stride (int): Stride size in the direction of the length.
        padding (int): Padding added at both sides in the direction of the length.
        dilation (int): Dilation size in the direction of the length.

    Returns:
        (str or int): The result of the operation.
    """
    operation_sympy = sympify_variable('1+(length+2*padding-dilation*(kernel-1)-1)/stride')
    output_sympy = operation_sympy.subs({'length': sympify_variable(length),
                                         'kernel': sympify_variable(kernel),
                                         'stride': sympify_variable(stride),
                                         'padding': sympify_variable(padding),
                                         'dilation': sympify_variable(dilation)})
    frac_sympy = output_sympy.subs({s: 0 for s in output_sympy.free_symbols})
    output_sympy = output_sympy - frac_sympy + numbers.Integer(frac_sympy)
    return get_nonrational_variable(output_sympy)
