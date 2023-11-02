# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Any, Dict, List, Tuple

from collections import defaultdict

import torch


def _make_single_op_gm(node) -> torch.fx.GraphModule:
    """Make a GraphModule that just executes the given node."""
    g = torch.fx.Graph()
    env = {}
    # TODO: Handle kwargs.
    assert not node.kwargs, "kwargs not supported yet"
    for arg in node.args:
        env[arg.name] = g.placeholder(arg.name)
    call = g.node_copy(node, lambda n: env[n.name])
    g.output(call)
    g.lint()
    return torch.fx.GraphModule(torch.nn.Module(), g)


def _identity_backend(gm: torch.fx.GraphModule,
                      example_inputs: List[torch.Tensor]):
    """A backend that just runs the given GraphModule as-is."""
    return gm


def _make_last_use_map(g: torch.fx.Graph) -> Dict[torch.fx.Node, List[torch.fx.Node]]:
    """Compute a map from each node to the last use of its value.

    Args:
        g: The graph to compute the last use map for.
    Returns:
        A map from each Node `n` to the set of Node's whose lifetime ends after
        `n` executes.
    """

    # Simple backward liveness analysis.
    # Iterate in reverse and when we see a use of a node for the first time,
    # that must be where its lifetime ends.
    seen = set()
    last_use_map = defaultdict(list)

    def process_use(user: torch.fx.Node, use: torch.fx.Node):
        if use not in seen:
            # Lifetime just ended, so this is the last use.
            seen.add(use)
            last_use_map[user].append(use)
    for node in reversed(g.nodes):
        assert not node.kwargs, "kwargs not supported yet"
        torch.fx.map_arg(node.args, lambda n: process_use(node, n))
    return last_use_map


def make_lockstep_debug_backend(golden_backend=_identity_backend):
    """Decorator that compares the wrapped backend to `golden_backend`.

    The returned wrapped backend is expected to be called with a GraphModule
    that only contains call_function nodes (call_module/call_method are not
    supported yet). This is the form of GraphModule's produced by `make_fx`
    and backends wrapped in `make_simple_dynamo_backend`.

    Currently, the behavior is to abort on the first mismatch and report
    the error.

    NOTE: The exact reporting and interface here is subject to change. Please
    try it out and provide feedback (or patches :) ).
    - make_fx should not drop source locations:
      https://github.com/pytorch/pytorch/issues/90276
    - Report tensors better (huge tensors should be summarized)
    - Maybe don't abort, but just warn?
    - Allow customizing atol/rtol.
    - How best to print the failing node? And include surrounding graph
      context?

    Args:
        golden_backend: A backend to compare the wrapped backend to. Defaults
        to eagerly executing the GraphModule op by op.
    Returns:
        A backend that compares the wrapped backend to `golden_backend`.
    """
    def wrapper(user_backend):
        def backend(gm: torch.fx.GraphModule,
                    example_inputs: List[torch.Tensor]):
            # We can ignore the example_inputs since we recompile in lockstep
            # anyway. TorchDynamo should already have appropriate guards in
            # place so that this doesn't change the compilation result.
            backend_artifacts: Dict[torch.fx.Node, Tuple[Any, Any]] = {}
            g = gm.graph
            last_use_map = _make_last_use_map(g)

            def compiled(*args):
                env = {}
                for placeholder, arg in zip([n for n in g.nodes if n.op == "placeholder"], args):
                    env[placeholder] = arg
                # Evaluate the graph one node at a time, comparing the user and
                # golden backends. This code currently does not support
                # get_attr and call_method/call_module due to it not being clear
                # how to best handle the recursion into submodules.
                # Thankfully, the graphs produced by make_fx obey this
                # restriction, so it is not a big deal.
                # TODO: Implement get_attr/call_method/call_module.
                for node in g.nodes:
                    if node.op == "placeholder":
                        # Already handled above.
                        continue
                    if node.op == "output":
                        return torch.fx.map_arg(node.args[0], lambda n: env[n])
                    assert node.op == "call_function", f"call_module/call_method not supported for {node} -- perhaps call make_simple_dynamo_backend first"
                    assert not node.kwargs, "kwargs not supported yet"
                    actual_args = torch.fx.map_arg(node.args, lambda n: env[n])
                    if node not in backend_artifacts:
                        # This will be populated on first run and will not need
                        # to recompile after.
                        gm = _make_single_op_gm(node)
                        backend_artifacts[node] = (
                            user_backend(gm, actual_args),
                            golden_backend(gm, actual_args),
                        )
                    user_compiled, golden_compiled = backend_artifacts[node]
                    user_result = user_compiled(*actual_args)
                    golden_result = env[node] = golden_compiled(*actual_args)
                    assert torch.allclose(user_result, golden_result), (
                        f"User result {user_result} is not close to "
                        f"golden result {golden_result} for "
                        f"node {node} at {node.stack_trace}")
                    # Clean up any tensors that are no longer needed.
                    # TODO: Find a way to test this.
                    # This was tested manually by printing the number of entries
                    # in `env` for a simple test case.
                    for dead_node in last_use_map[node]:
                        env.pop(dead_node)
                assert False, "not reached -- missing 'output' node"
            return compiled
        return backend
    return wrapper
