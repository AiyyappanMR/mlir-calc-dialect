# export all of our auto-generated python wrapper classes
from ._calc_ops_gen import *

# register our custom passes with the mlir pass registry
import calc_dialect._calcPasses  
calc_dialect._calcPasses.register_calc_passes()
