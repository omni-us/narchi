
from .propagators.simple import propagators as simple_propagators
from .propagators.same import propagators as same_propagators
from .propagators.conv import propagators as conv_propagators
from .propagators.pool import propagators as pool_propagators
from .propagators.rnn import propagators as rnn_propagators
from .propagators.reshape import propagators as reshape_propagators
from .propagators.group import propagators as group_propagators
from .module import propagators as module_propagators


module_propagators = [simple_propagators,
                      same_propagators,
                      conv_propagators,
                      pool_propagators,
                      rnn_propagators,
                      reshape_propagators,
                      group_propagators,
                      module_propagators]

propagators = {}


def register_propagator(propagator, replace=False):
    """Adds a propagator to the dictionary of registered propagators."""
    if not replace and propagator.block_class in propagators:
        raise ValueError('Propagator for blocks of type '+propagator.block_class+' already registered.')
    propagators[propagator.block_class] = propagator


def register_package_propagators():
    """Function that registers all propagators defined in the modules of the package."""
    for module in module_propagators:
        for propagator in module:
            register_propagator(propagator)


register_package_propagators()
