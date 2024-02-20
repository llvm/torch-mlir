# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
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
from typing import Any, Callable, List, NamedTuple, Optional, TypeVar, Union, Dict
from itertools import repeat

import os
import sys
import traceback

import multiprocess as mp
from multiprocess import set_start_method
try:
    set_start_method("spawn")
except RuntimeError:
    # Children can error here so we suppress.
    pass

import torch

TorchScriptValue = Union[int, float, List['TorchScriptValue'],
                         Dict['TorchScriptValue',
                              'TorchScriptValue'], torch.Tensor]


class TraceItem(NamedTuple):
    # The externally visible symbol name that is called.
    # For example `"forward"` or `"submodule.forward"`.
    symbol: str
    # The inputs to the call.
    inputs: List[TorchScriptValue]
    # The output from the call.
    # In Python, there is only one output from a function. It might be a tuple
    # in case of "multiple results".
    # Sometimes this field is treated as golden outputs from a test.
    # Sometimes this field is treated as ignored, such as the input trace
    # provided to `TestConfig.run`.
    output: TorchScriptValue


# A trace of invocations to the program.
# This is an ordered sequence of external invocations to a program's
# public boundary.
Trace = List[TraceItem]


# Clone all the tensor values.
def clone_torch_script_value(v: TorchScriptValue):
    if isinstance(v, torch.Tensor):
        return v.clone()
    if isinstance(v, tuple):
        return tuple(clone_torch_script_value(field) for field in v)
    if isinstance(v, list):
        return [clone_torch_script_value(item) for item in v]
    if isinstance(v, dict):
        return {
            clone_torch_script_value(key): clone_torch_script_value(val)
            for key, val in v.items()
        }
    if isinstance(v, float) or isinstance(v, int) or isinstance(v, str):
        return v
    assert False, "unhandled cloning of TorchScriptValue value type"


# This clone helper is used to work around issues with output tensors when
# using multiprocessing module to run tests. The error happens for tests like
# ContiguousModule_basic where the output tensor aliases with an input tensor.
# When the output tensor is not cloned, the testing trace would be modified for
# unknown reason when passed through the shared memory through synchronized
# queue for example.
# TODO: Figure out the root cause of the failure and fix properly.
def clone_trace(trace: Trace) -> Trace:
    return [
        TraceItem(symbol=item.symbol,
                  inputs=clone_torch_script_value(item.inputs),
                  output=clone_torch_script_value(item.output))
        for item in trace
    ]


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
    Backends to torch-mlir will likely have more elaborate TestConfig's, such
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
    # We can have a helper class LinalgOnTensorsBackendTestConfig which does that.
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
    def rand(self, *sizes, low=0.0, high=1.0):
        return torch.empty(sizes).uniform_(low, high)

    def randint(self, *sizes, low=0, high=10, dtype=torch.int64):
        return torch.randint(low, high, sizes, dtype=dtype)

    def nans(self, *sizes):
        vals = torch.empty(sizes)
        vals[...] = torch.nan
        return vals


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
    # If runtime failed, a string describing the failure.
    # If this is not None, then the `trace` and `golden_trace` fields are None,
    # and vice-versa.
    runtime_error: Optional[str]
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
        # Clone the inputs to capture the original tensors values. This is
        # needed because inplace mutation might happen to the input tensors.
        inputs = [clone_torch_script_value(arg) for arg in args]
        output = self.__wrapped__(*args, **kwargs)
        self.__trace__.append(
            TraceItem(symbol=".".join(self.__property_base_path__),
                      inputs=inputs,
                      output=output))
        return output

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


def compile_and_run_test(test: Test, config: TestConfig, verbose=False) -> Any:
    try:
        golden_trace = generate_golden_trace(test)
        if verbose:
            print(f"Compiling {test.unique_name}...", file=sys.stderr)
        compiled = config.compile(test.program_factory())
    except Exception as e:
        return TestResult(unique_name=test.unique_name,
                          compilation_error="".join(
                              traceback.format_exception(
                                  type(e), e, e.__traceback__)),
                          runtime_error=None,
                          trace=None,
                          golden_trace=None)
    try:
        if verbose:
            print(f"Running {test.unique_name}...", file=sys.stderr)
        trace = config.run(compiled, golden_trace)
    except Exception as e:
        return TestResult(unique_name=test.unique_name,
                          compilation_error=None,
                          runtime_error="".join(
                              traceback.format_exception(
                                  type(e), e, e.__traceback__)),
                          trace=None,
                          golden_trace=None)
    return TestResult(unique_name=test.unique_name,
                      compilation_error=None,
                      runtime_error=None,
                      trace=clone_trace(trace),
                      golden_trace=clone_trace(golden_trace))


def run_tests(tests: List[Test], config: TestConfig, sequential=False, verbose=False) -> List[TestResult]:
    """Invoke the given `Test`'s with the provided `TestConfig`."""
    num_processes = min(int(mp.cpu_count() * 0.8) + 1, len(tests))
    try:
        env_concurrency = int(os.getenv("TORCH_MLIR_TEST_CONCURRENCY", "0"))
    except ValueError as e:
        raise ValueError("Bad value for TORCH_MLIR_TEST_CONCURRENCY env var: "
                         "Expected integer.") from e
    if env_concurrency > 0:
        num_processes = min(num_processes, env_concurrency)

    # TODO: We've noticed that on certain 2 core machine parallelizing the tests
    # makes the llvm backend legacy pass manager 20x slower than using a
    # single process. Need to investigate the root cause eventually. This is a
    # hack to work around this issue.
    # Also our multiprocessing implementation is not the most efficient, so
    # the benefit at core count 2 is probably not worth it anyway.
    if mp.cpu_count() == 2:
        num_processes = 1

    # Sort the tests to make output nicer.
    tests = list(sorted(tests, key=lambda t: t.unique_name))

    # TODO: If num_processes == 1, then run without any of the multiprocessing
    # machinery. In theory it should work, but any crash in the testing process
    # seems to cause a cascade of failures resulting in undecipherable error
    # messages.
    if num_processes == 1 or sequential:
        return [compile_and_run_test(test, config, verbose) for test in tests]

    # This is needed because autograd does not support crossing process
    # boundaries.
    torch.autograd.set_grad_enabled(False)

    pool = mp.Pool(num_processes)
    arg_list = zip(tests, repeat(config))
    handles = pool.starmap_async(compile_and_run_test, arg_list)
    results = handles.get(timeout=360)

    tests_with_results = {result.unique_name for result in results}
    all_tests = {test.unique_name for test in tests}
    # For processes that are crashed due to compile time or runtime error,
    # the error outputs are printed out all together but no TestResult is
    # produced when the process crashed.
    # TODO: Find a clean way to capture the output from crashed process and
    # create more detailed runtime_error for those tests.
    aborted_tests = all_tests - tests_with_results
    aborted_tests_results = [
        TestResult(
            unique_name=aborted_test_name,
            compilation_error=None,
            runtime_error=
            "Testing process terminated. Either the compiler crashed or the compiled code crashed at runtime.\n",
            trace=None,
            golden_trace=None) for aborted_test_name in aborted_tests
    ]
    results.extend(aborted_tests_results)
    results.sort(key=lambda result: result.unique_name)
    return results
