# Roadmap as of beginning of 2021Q2

## Non-technical project status overview

The project has evolved past "works on my machine" stage. It's hard to provide
meaningful numbers for an open-source project, but I'm seeing >5 people
regularly active on pull requests, bugs, etc. covering aspects ranging from
acap_dispatch, TorchScript, RefBackend, build systems, and even CI. This is very
promising and feels healthy to me.

## Roadmap overview

The project has grown a number of aspects, but effort has converged on 3
workstreams:

- acap_dispatch: The goal of this project is to develop a tracing-based frontend
  for Torch interoperability that takes clues from existing working solutions to
  enable a gateway from PyTorch to MLIR.
  - Why this project is cool: For users that can tolerate the limitations of
    tracing systems, this project enables an MLIR-based frontend for PyTorch on
    a shorter-time frame than the TorchScript compilation, letting downstream
    users focus on their value-add. Also, the tracing-based approach has a
    distinct usability advantage for many pure-Python researcher workflows.

- TorchScript compilation: The goal of this project is to build the frontend of
  a truly next-generation ahead-of-time machine learning compiler.
  - Why this project is cool: This system is designed from day 1 to support
    features such as dynamic shapes, control flow, mutable variables,
    program-internal state, and non-Tensor types (scalars, lists, dicts) in a
    principled fashion. These features are essential for empowering an
    industry-level shift in the set of machine learning programs that are
    feasible to deploy with minimal effort across many devices (when combined
    with a backend using the advanced compilation techniques being developed
    elsewhere in the MLIR ecosystem).

- Reference backend (RefBackend): The goal of this project is to develop a
  reference end-to-end flow for the MLIR project, using the needs of our
  frontends as seeds for new feature development and upstreaming.
  - Why this project is cool: Due to its status as an LLVM incubator project,
    npcomp is uniquely positioned to develop an end-to-end flow with a clear
    path toward upstreaming components as their design converges (example:
    bufferization), or rapidly rebasing on newly added upstream components to
    replace homegrown pieces (example: linalg on tensors).

## acap_dispatch

acap_dispatch is the name of our implementation of a tracing-based PyTorch
program capture system, analogous to the one used by
[pytorch/xla](https://github.com/pytorch/xla). This system is sufficient to
capture very many programs of interest, and has the benefit of seamlessly
capturing gradients, shapes, and dtypes, while still bottoming out on the same
ATen dialect needed by the TorchScript path. It also trivializes all use of
Python-data structures like lists by directly observing their values as
constants.

Looking a bit longer-term, this flow is a good complement to the TorchScript
flow and has distinct tradeoffs. These are captured nicely in the paper
[LazyTensor: combining eager execution with domain-specific
compilers](https://arxiv.org/abs/2102.13267). In their terminology, our
acap_dispatch path implements "Tracing" while our TorchScript path implements
"Direct compilation". Direct compilation tends to be required for deploying
complex models for inference, edge, or federated learning applications, while
Tracing is the building block for a totally seamless researcher experience when
iterating in Python.

### 2021Q2

- Improve robustness of the flow's program capture, ideally to the level of
  `pytorch/xla`.
- Get into a steady-state where adding operations is fairly mechanical.
- Support at least a few programs of community interest.
- Identify demand for a more holistic user experience, analogous to
  `pytorch/xla`. For example, building out support for the more runtime-y
  aspects like compiling on the fly, moving tensors in and out of the compiler
  system's runtime, etc. that makes it an actual user experience rather than
  just a way to get compiler IR.

## TorchScript compilation

The TorchScript compiler represents the bulk of core compiler effort in the
npcomp project.
[TorchScript](https://pytorch.org/docs/stable/jit_language_reference.html) is a
restricted (more static) subset of Python, but even TorchScript is quite dynamic
when compared to the needs of lower-levels of the compilation stack, especially
systems like Linalg. The overarching theme of this project is building out
compiler components that bridge that gap. As we do so, the recurring tradeoffs
are:

- user experience: we want a fairly unrestricted programming model -- that's
  what users like about PyTorch, and what enables users to deploy without
  significant modifications of their code.
- feasibility of the compiler: we want a smart compiler that is feasible to
  implement (for our own sanity :) )
- excellent generated code quality: this is of course dependent on the backend
  which is paired with the frontend we are building, but there are a number of
  transformations that make sense before we reach the backend which strongly
  affect the quality of code generated from a backend.

To give a concrete example, consider the problem of inferring the shapes of
tensors at various points in the program. The more precision we have on the
shapes, the better code can be emitted by a backend. But in general, users need
to provide at least some information about their program to help the compiler
understand what shapes are at different points in the program. The smarter our
compiler algorithms are, the less information the user needs to provide. Thus,
all 3 facets are interlinked and there is no single right answer -- we need to
balance them for a workable system.

To accomplish this goal, we intend to be guided by a *model curriculum*, which
consists of programs of escalating complexity, from a simple elementwise
operation all the way to a full-blown end-to-end speech recognition program. Our
development process consists of setting incremental objectives to build out new
layers of the compiler to a satisfactory level on easier programs in the
curriculum, and backfilling complexity as needed to extend to the harder
programs. Ideally, this backfilling does not require deep conceptual changes to
components, but is simply an application of extension points anticipated in the
original design. The trick to making that happen is evaluating designs on enough
programs from the curriculum to ensure that a solution is likely to generalize
and satisfy our objectives, without getting bogged down in theoretical details.

### 2021Q2

- Model curriculum
  - Formalize / publish curriculum to ease collaboration
  - Incorporate end-to-end ASR (speech recognition) model into curriculum, or
    program of similar complexity.
  - Incorporate representative quantized models into curriculum.
- End-to-end execution of at least the simplest models in the curriculum.
  - User annotation infrastructure for users to provide the compiler
    information, such as shapes to seed shape inference.
  - Fill out ATen dialect and `aten-recognize-kernels` pass.
  - ATen lowering to Linalg-on-tensors
    - Implement a minimal amount of linalg-inspired abstractions in the "TCF"
      dialect.
    - Extend the linalg
      [OpDSL tooling](https://llvm.discourse.group/t/rfc-linalg-opdsl/2966/6) to
      enable us to programmatically emit shape validity checks.
  - Shape/dtype inference
    - As needed for other incremental objectives.
    - Build a clear picture of the right place(s) in the longer-term compiler
      pipeline for shape inference.
  - Canonicalizations and general compiler optimizations
    - As needed for other incremental objectives.
  - Backend choice: RefBackend or IREE candidates.

### 2021Q3

- Start to smell a little production-ey
  - For the simplest models at least, get them running on IREE with performance
    competitive with other frontends.
  - Write initial "user manual" (and any supporting tools) for how to use the
    new frontend (+ backend integration points) to deploy something.
- Extend model support:
  - Vertically integrated spike building out generalized support for list, dict,
    etc. for representative complex models. (co-design with RefBackend or IREE).
  - Implement coherent shape/dtype inference design based on Q2 insights.
- Scale up of Q2 compiler features to the curriculum
  - Extend user annotation infrastructure as needed.
  - ATen dialect and `aten-recognize-kernels` pass
  - ATen --> Linalg lowerings.
  - Canonicalizations and other compiler optimizations

## RefBackend

The npcomp reference backend (or "RefBackend") is perhaps the most confusing
part of the project, since it really has nothing to do per se with compiling
numerical Python programs. The RefBackend's biggest impact is really a strategic
play on two time horizons:

- short-medium term: Avoid bad design decisions by avoiding single-sourcing on
  IREE.
  - Although some key contributors to npcomp are closely affiliated with IREE,
    there is a distinct desire to honor the spirit of being an LLVM incubator
    and not have the npcomp project evolve into an extension of IREE. We also
    believe that this kind of design influence results in a better system in
    general.
- medium-long term: Give upstream MLIR a more "batteries included" end to end
  flow by incubating minimally-opinionated components and upstreaming them.
  - Context: Due to history, all MLIR-based end-to-end flows of nontrivial
    capability live in downstream repositories, such as TensorFlow, IREE, etc.
    This leads to an awkward situation where sometimes code is added to
    upstream, but any load-bearing use case cannot be exercised with upstream
    tools (such as quantifying performance, building auto-tuning infrastructure,
    etc.). This leads to significant drag on MLIR's overall trajectory.

The way we intend to advance those two strategic goals there is to incorporate
end-to-end execution on the RefBackend as part of the end-to-end execution
milestones of the acap_dispatch and TorchScript frontends.

### 2021Q2

- Build out support for PyTorch kernel fallback.
- Help Nicolas build out and ideally land upstream his linalg-on-tensors
  [e2e execution sandbox](https://github.com/google/iree/tree/main/experimental/runners),
  with an eye towards rebasing aspects of the RefBackend flow on those
  components.
- Build out better runtime calling convention interop.
- Start thinking about a plan to support list, dict, etc. in the runtime,
  ideally using MLIR infra to make it magically generalize and be minimally
  opinionated.

### 2021Q3

- Using the runtime abstractions built out for list, dict, etc., ditch the
  `memref`-based lowering flow and use new primitives for the "top-level" of the
  program (use of memref should be isolated from e.g. top-level control flow,
  lifetime management, calling conventions, etc.).
- Use (or help build) upstream linalg-on-tensors abstractions analogous to
  IREE's `flow.dispatch.workgroups` (parallel computation grid) that
  linalg-on-tensors can directly fuse into, avoiding phase ordering issues with
  fusion, bufferization, kernel generation.
