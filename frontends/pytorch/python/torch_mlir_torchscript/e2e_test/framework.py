#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
# End-to-end testing framework for TorchScript.

For the purposes of this framework, "end to end" means the first "end" is
a `torch.nn.Module`, and the second "end" is execution.

## Architecture

A program for this testing framework is considered to be a `torch.nn.Module`,
which has a public interface consisting of its methods and instance attributes.

A test in the framework consists conceputally of a list of calls into
the methods of a module (TODO: extend to instance attributes). It is expected
that the outputs match between the program run on a backend (controlled by
a TestConfig) and a golden trace obtained by running on native Torch (without
compiling or TorchScript'ing).
"""

import abc
from typing import Any, Callable, List, NamedTuple, Optional, TypeVar

import io
import pickle

import torch

from ..annotations import apply_serializable_annotations


class TraceItem(NamedTuple):
    # The externally visible symbol name that is called.
    # For example `"forward"` or `"submodule.forward"`.
    symbol: str
    # The list of inputs to the call.
    inputs: List[torch.Tensor]  # TODO: Support more types.
    # The outputs from the call.
    # Sometimes this field is treated as golden outputs from a test.
    # Sometimes this field is treated as ignored, such as the input trace
    # provided to `TestConfig.run`.
    outputs: List[torch.Tensor]  # TODO: Support more types


# A trace of invocations to the program.
# This is an ordered sequence of external invocations to a program's
# public boundary.
Trace = List[TraceItem]

# A type shared between the result of `TestConfig.compile` and the input
# to `TestConfig.run`. Each backend will likely have a different definition of
# this type.
CompiledArtifact = TypeVar('CompiledArtifact')


class TestConfig(abc.ABC):
    """The interface implemented by backends to run tests.

    The testing framework expects to be able to call `compile` to compile
    a torch.nn.Module, and then pass the compiled artifact to `run` to run it.

    Note that the definition of "compiled artifact" here is quite loose, and
    this interface allows for many different use cases besides simple testing.

    For example, this interface can be overridden to be a "data collector"
    to gather information across all the test cases. For example,
    a compiler backend could override "compile" to just return some IR at a
    useful intermediate abstraction level (rather than the final compiled
    artifact), and then have "run" save this intermediate IR + the trace as
    input to some lower-level software stack's testing format.

    The set of TestConfig's is expected to be pluggable and provided by
    users to suit their own needs. We provide a few configs out of the box
    in the `configs` submodule of this package, but those are intended
    to be for basic inspiration and enough for our own testing.
    Backends to npcomp will likely have more elaborate TestConfig's, such
    as `compile` being "compile for such-and-such DSP with these vectorization
    cost model flags" and `run` being "connect to Android phone with
    device ID 1234 and upload a program to run on it's DSP core, and also set
    power throttling settings to 'performance'".

    That is also why this class is not called "backend", as it
    encapsulates potentially many specific details of the test configuration
    process as well. There isn't a general way to disentangle test configuration
    from the compile/run process specific to a logical backend, since each
    backend (compiler backend and runtime target) will have an arbitrarily
    wild and wonderful set of possible configurations that we cannot predict.
    """
    # This is not a frontend-lowered module, to allow various testing at the PyTorch level.
    # We can have a helper class NpcompBackendTestConfig which does that.
    @abc.abstractmethod
    def compile(self, program: torch.nn.Module) -> CompiledArtifact:
        """Compile the provided torch.nn.Module into a compiled artifact"""
        pass

    # Any should match result of `compile`.

    @abc.abstractmethod
    def run(self, artifact: CompiledArtifact, trace: Trace) -> Trace:
        """Run the compiled artifact produced by `compile`.

        The backend should load the compiled artifact and call the
        symbol names listed in `trace` with their respective inputs (the outputs
        of `trace` should be ignored). A new identical trace with outputs
        populated should be returned.

        This method should assume that `artifact` is being shared with
        multiple parallel invocations of `run`, and so it should not be mutated.
        This property is typicaly trivially satisfied for a true
        "compiled artifact", but some backends don't directly involve a
        compiled artifact per se (like a backend for which `CompiledArtifact` is
        `torch.nn.Module` and `run` just invokes the torch.nn.Module itself)

        Args:
            artifact: a compiled artifact produced by `compile`.
            trace: The external invocations to stimulate the module.
        Returns:
            A trace with outputs recorded according to the results of running
            on this backend.
        """
        pass


# Utilities for common testing trace generation.
# Also, resets the random seed for reproducibility.
# TODO: If generating in parallel, how to have manual_seed be local?
class TestUtils:
    """Utilities for executing a test.

    Test cases are provided an instance of this class to make test cases
    more succinct.

    For reproducibility, this class also resets the random seed.
    TODO: Figure out how to seed reset properly scoped to just a test case
    (such as when running tests in parallel)
    """
    def __init__(self):
        torch.manual_seed(0)

    # TODO: Add zeros/ones/etc. as convenient.
    def rand(self, *sizes):
        if len(sizes) == 0:
            return torch.rand([])
        return torch.rand(*sizes)


class Test(NamedTuple):
    """A description of a test as produced by the test frontend.
    """
    # Stable name for error reporting.
    #
    # This name's stability is also useful for backend, which want to
    # generate their own lower-level test suites based on this framework.
    #
    # It is expected that those backends will need additional
    # metadata to describe their test configurations, so having a unique
    # key to keep that information associated is important.
    unique_name: str
    # A callable which produces the module under test.
    # This is a callable to allow lazily creating the module.
    program_factory: Callable[[], torch.nn.Module]
    # A callable which provides external stimuli to the module.
    # The first parameter is a torch.nn.Module (or a `_Tracer` wrapping that
    # module, actually).
    # The secon parameter is a `TestUtils` instance for convenience.
    program_invoker: Callable[[Any, TestUtils], None]


class SerializableTest(NamedTuple):
    """A self-contained representation of a test that can be pickled.

    We use serialized TorchScript programs here for two reasons:
    1. The PyTorch pickling story isn't great, so in order to reliably pickle
       this class, we rely on having the serialized bytes for the TorchScript
       module already given to us.
    2. The choice of a TorchScript module vs `torch.nn.Module` boils down to
       the fact that `torch.nn.Module` cannot be deserialized without pulling
       in the same set of Python dependencies that were used to serialize it
       in the first place. This would defeat one of the
       main use cases of this class, which is to transport a test from an
       environment with a set of heavy dependencies to a dependency-light one.
       Since TorchScript modules are self-contained, they fit the bill
       perfectly.
    """
    # See unique_name on `Test`.
    unique_name: str
    # Serialized TorchScript program.
    program: bytes
    # Trace for execution testing.
    trace: Trace

    def as_test(self) -> Test:
        """Create a `Test` from this class."""
        # Conform the serialized program to the interface expected by Test.
        # This is a bit of a hack, but it's the only way to keep the layering
        # straight.
        def factory():
            _extra_files = {"annotations.pkl": ""}
            module = torch.jit.load(io.BytesIO(self.program),
                                    _extra_files=_extra_files)
            # Load the pickled annotations.
            annotations = pickle.loads(_extra_files["annotations.pkl"])
            apply_serializable_annotations(module, annotations)
            return module

        def invoker(module, tu):
            for item in self.trace:
                attr = module
                for part in item.symbol.split("."):
                    attr = getattr(attr, part)
                attr(*item.inputs)

        return Test(
            unique_name=self.unique_name,
            program_factory=factory,
            program_invoker=invoker,
        )


class TestResult(NamedTuple):
    # Stable unique name for error reporting and test suite configuration.
    #
    # Tests frequently need some additional data (such as expected pass/fail
    # status, desired test configurations, etc.), and this gives a key to
    # associate to. This avoids extending this class arbitrarily for every
    # possible requirement from the test framework.
    #
    # This name is also useful for backends that are generating their own
    # lower-level test suites from this framework for the same reasons, though
    # those reasons are stronger because we cannot simply extend this
    # class.
    unique_name: str  # Should match Test.unique_name for corresponding test.
    # If compilation failed, a string describing the failure.
    # If this is not None, then the `trace` and `golden_trace` fields are None,
    # and vice-versa.
    compilation_error: Optional[str]
    # The trace produced by the backend.
    trace: Optional[Trace]
    # The golden trace which `trace` is expected to match.
    golden_trace: Optional[Trace]


class _Tracer:
    """Wrapper around a `torch.nn.Module` that records calls into it.

    The inputs and outputs of each call are recorded in a Trace. Recursive
    property accesses are also traced.
    """
    def __init__(self, wrapped, property_base_path: List[str], trace: Trace):
        self.__wrapped__ = wrapped
        self.__trace__ = trace
        self.__property_base_path__ = property_base_path

    def __call__(self, *args, **kwargs):
        raw_outputs = self.__wrapped__(*args, **kwargs)
        if isinstance(raw_outputs, torch.Tensor):
            outputs = [raw_outputs]
        elif isinstance(raw_outputs, tuple) and all(
                isinstance(o, torch.Tensor) for o in raw_outputs):
            outputs = raw_outputs
        else:
            raise Exception("unimplemented: non-Tensor output from function")
        self.__trace__.append(
            TraceItem(symbol=".".join(self.__property_base_path__),
                      inputs=args,
                      outputs=outputs))
        return raw_outputs

    def __getattr__(self, name):
        return _Tracer(getattr(self.__wrapped__, name),
                       self.__property_base_path__ + [name], self.__trace__)


def generate_golden_trace(test: Test) -> Trace:
    """Generate a trace with the original program.

    If the original program is deterministic, then this the produced trace is
    suitable as a golden trace to compare against.
    """
    trace = []
    tracer = _Tracer(test.program_factory(), [], trace)
    test.program_invoker(tracer, TestUtils())
    return trace


def run_tests(tests: List[Test], config: TestConfig) -> List[TestResult]:
    """Invoke the given `Test`'s with the provided `TestConfig`."""
    results = []
    for test in tests:
        # TODO: Precompile everything in parallel.
        try:
            golden_trace = generate_golden_trace(test)
            compiled = config.compile(test.program_factory())
        except Exception as e:
            # Useful for debugging:
            # ```
            # raise
            # ```
            # This will give the full traceback rather than giving just
            # the stringified exception in the report.
            # TODO: Capture the traceback and make it available in the report.
            results.append(
                TestResult(unique_name=test.unique_name,
                           compilation_error=str(e),
                           trace=None,
                           golden_trace=None))
            continue
        # TODO: Run in parallel.
        trace = config.run(compiled, golden_trace)
        results.append(
            TestResult(unique_name=test.unique_name,
                       compilation_error=None,
                       trace=trace,
                       golden_trace=golden_trace))
    return results
