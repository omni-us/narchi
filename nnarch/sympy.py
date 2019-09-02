"""Functions for symbolic operations."""

import re
import sympy
from sympy.core import numbers
from .schema import variable_pattern


variable_regex = re.compile('^'+variable_pattern+'$')


def variable_operate(input, operation):
    """Performs a symbolic operation on a given input.

    Args:
        input (str or int): Input to operate on, either an int or a variable, e.g. "<<variable:W/2+H/4>>".
        operation (str or int): Operation to apply on input, either int or expression, e.g. "__input__/3".

    Returns:
        (str or int): The result of the operation.

    Raises:
        ValueError:
            * If operation is not int or not a valid expression containing only "__input__" free symbol.
            * If input is not an int or a string that follows variable_regex.pattern.
            * If input is not a valid expression or contains "__input__" as a free symbol.
    """
    ## Handle operation ##
    if isinstance(operation, int):
        return operation
    elif not isinstance(operation, str):
        raise ValueError('Expected operation to be an int or a string.')
    operation_sympy = sympy.sympify(operation)
    operation_symbs = {str(s) for s in operation_sympy.free_symbols}
    if not (operation_symbs == {'__input__'} or len(operation_symbs) == 0):
        raise ValueError('Operation can only contain "__input__" as a free symbol, got operation='+operation)
    ## Handle input ##
    var_match = variable_regex.match(str(input))
    if isinstance(input, int):
        input_sympy = sympy.sympify(input)
    elif var_match:
        input_sympy = sympy.sympify(var_match[1])
    else:
        raise ValueError('Expected input to be an int or a string with pattern: '+variable_regex.pattern)
    input_symbs = {str(s) for s in input_sympy.free_symbols}
    if '__input__' in input_symbs:
        raise ValueError('Input must not contain "__input__" as a free symbol, got input='+input)
    ## Substitute input into operation ##
    output_sympy = operation_sympy.subs({'__input__': input_sympy})
    ## Return operation result ##
    if isinstance(output_sympy, numbers.Integer):
        return int(output_sympy)
    return '<<variable:'+str(output_sympy).replace(' ', '')+'>>'
