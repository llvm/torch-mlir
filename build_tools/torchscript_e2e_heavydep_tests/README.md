# Additional TorchScript end-to-end tests with heavy dependencies
Some of the Torchscript end-to-end tests require additional dependencies which don't make sense to include as part of the default torch-mlir setup. Additionally, these dependencies often don't work with the same HEAD PyTorch version that torch-mlir builds against at the C++ level (the TorchScript importer is written in C++)

We have a self-contained script that generates all the needed artifacts from a self-contained virtual environment. It can be used like so:

# Build the virtual environment in the specified directory and generate the
# serialized test artifacts in the other specified directory.
# This command is safe to re-run if you have already built the virtual
# environment and just changed the tests.
build_tools/torchscript_e2e_heavydep_tests/generate_serialized_tests.sh \
  path/to/heavydep_venv \
  path/to/heavydep_serialized_tests

# Add the --serialized-test-dir flag to point at the directory containing the
# serialized tests. All other functionality is the same as the normal invocation
# of torchscript_e2e_test.sh, but the serialized tests will be available.
tools/torchscript_e2e_test.sh --serialized-test-dir=path/to/heavydep_serialized_tests
The tests use the same (pure-Python) test framework as the normal torchscript_e2e_test.sh, but the tests are added in build_tools/torchscript_e2e_heavydep_tests instead of frontends/pytorch/e2e_testing/torchscript.

We rely critically on serialized TorchScript compatibility across PyTorch versions to transport the tests + pure-Python compatibility of the torch API, which has worked well so far.
