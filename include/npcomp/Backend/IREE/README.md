# IREE Backend

Passes/utilities for preparing input to IREE.

For now, this directory doesn't have a C++-level dependency on IREE, since
it only performs a trivial transformation. Eventually, if lowering
nontrivial constructs to IREE (such as a list type to `!iree.list`),
we will need to take that dependency, and it will be isolated to this directory.
