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
from typing import Any, Callable, List, NamedTuple, TypeVar

import torch


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
    # The trace produced by the backend.
    trace: Trace
    # The golden trace which `trace` is expected to match.
    golden_trace: Trace


class _Tracer:
    """Wrapper around a `torch.nn.Module` that records calls into it.

    The inputs and outputs of each call are recorded in a Trace.
    """
    module: torch.nn.Module
    trace: Trace

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.trace = []

    def __getattr__(self, name):
        # TODO: Handle `module.foo.bar.baz` nesting.
        # For now, we are limited to attributes of the top-level module.
        def invoke(*args):
            raw_outputs = getattr(self.module, name)(*args)
            if isinstance(raw_outputs, torch.Tensor):
                outputs = [raw_outputs]
            self.trace.append(
                TraceItem(symbol=name, inputs=args, outputs=outputs))
            return raw_outputs

        return invoke

    def get_trace(self):
        return self.trace


def _generate_golden_trace(test: Test) -> Trace:
    tracer = _Tracer(test.program_factory())
    test.program_invoker(tracer, TestUtils())
    return tracer.get_trace()


def run_tests(tests: List[Test], config: TestConfig) -> List[TestResult]:
    """Invoke the given `Test`'s with the provided `TestConfig`."""
    results = []
    for test in tests:
        golden_trace = _generate_golden_trace(test)
        # TODO: Precompile everything in parallel.
        compiled = config.compile(test.program_factory())
        # TODO: Run in parallel.
        trace = config.run(compiled, golden_trace)
        results.append(
            TestResult(unique_name=test.unique_name,
                       trace=trace,
                       golden_trace=golden_trace))
    return results


def report_results(results: List[TestResult]):
    """Provide a basic error report summarizing various TestResult's."""
    for result in results:
        failed = False
        for item_num, (item, golden_item) in enumerate(
                zip(result.trace, result.golden_trace)):
            assert item.symbol == golden_item.symbol
            assert len(item.inputs) == len(golden_item.inputs)
            assert len(item.outputs) == len(golden_item.outputs)
            for input, golden_input in zip(item.inputs, golden_item.inputs):
                assert torch.allclose(input, golden_input)
            for output_num, (output, golden_output) in enumerate(
                    zip(item.outputs, golden_item.outputs)):
                # TODO: Refine error message. Things to consider:
                # - Very large tensors -- don't spew, but give useful info
                # - Smaller tensors / primitives -- want to show exact values
                # - Machine parseable format?
                if not torch.allclose(output, golden_output):
                    print(
                        f'Error: in call #{item_num} into the module: result #{output_num} not close in call to "{item.symbol}"'
                    )
                    failed = True
        if failed:
            print('FAILURE "{}"'.format(result.unique_name))
        else:
            print('SUCCESS "{}"'.format(result.unique_name))
