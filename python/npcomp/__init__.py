# TODO: Remove this hack once switching to the upstream python bindings.
# In the new world, no python extensions directly contain MLIR C++ code (which
# needs global linkage). The legacy bindings do and directly have vague
# linkage on symbols like:
#   mlir::detail::TypeIDExported::get<mlir::FuncOp>()::instance
# (which must be a static singleton)
# Forcing global linkage of the _npcomp extension works around the issue
# until that code can be excised.
def _load_extension():
  import sys
  import ctypes
  flags = sys.getdlopenflags()
  sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)
  import _npcomp
  sys.setdlopenflags(flags)


_load_extension()
