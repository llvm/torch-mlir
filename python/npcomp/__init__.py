from mlir import _cext_loader
_cext_loader._cext.globals.append_dialect_search_prefix("npcomp.dialects")

_cext = _cext_loader._load_extension("_npcomp")
_cext._register_all_passes()
_cext._initialize_llvm_codegen()

# Top-level symbols.
from .exporter import *
from .types import *

from . import tracing
from . import utils
