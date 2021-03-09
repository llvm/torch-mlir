def _load_extension():
  # TODO: Remote the RTLD_GLOBAL hack once local, cross module imports
  # resolve symbols properly. Something is keeping the dynamic loader on
  # Linux from treating the following vague symbols as the same across
  # _mlir and _npcomp:
  #   mlir::detail::TypeIDExported::get<mlir::FuncOp>()::instance
  import sys
  import ctypes
  flags = sys.getdlopenflags()
  sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)
  import _npcomp
  sys.setdlopenflags(flags)

  from mlir._cext_loader import _cext
  _cext.globals.append_dialect_search_prefix("npcomp.dialects")
  return _npcomp


_cext = _load_extension()
_cext._register_all_passes()
_cext._initialize_llvm_codegen()

# Top-level symbols.
from .exporter import *
from .types import *

from . import tracing
from . import utils
