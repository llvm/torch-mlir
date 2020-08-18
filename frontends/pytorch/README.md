# NPComp - PyTorch frontend integration

This directory contains optional components for interfacing PyTorch to NPComp.
Integration is targeted at multiple levels:

* Via program capture with a ATen pseudo-device.
* Via IR-level integration with PyTorch (via tracing or scripting interfaces).
* Interfaces to facilitate checking against reference implementations and
  verification.

In all situations, the target dialects are maintained in the outer project,
along with their lowerings to common intermediate dialects and backends. This
directory should be purely about interfacing with the PyTorch/LibTorch
components for extracting and executing programs.

See the [overall documentation for frontends](../README.md) for further details
about code layout and integration philosophy.
