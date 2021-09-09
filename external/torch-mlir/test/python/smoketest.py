# RUN: %PYTHON %s
# XXX: Fix this
# XFAIL: *

import mlir.ir
from mlir.dialects import iree

with mlir.ir.Context() as ctx:
  iree.register_iree_dialect(ctx)
