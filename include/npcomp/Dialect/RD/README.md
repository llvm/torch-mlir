# Routine Data #

A co-Routine-based implementation of Data pipelines.

> Current status: under development (completely broken).

## Op Overview ##

There are a few kinds of ops in the RD dialect:

 - **Dataset Ops**: These ops create datasets or transform a dataset into a new dataset.
   These ops are used to declaratively describe the computation to be performed, and are
   optimized away at compile time.
 - **Iterator Ops**: These ops define the interface for accessing elements within a RD
   dataset by using iterators. These ops are called by the user's program.
 - **Internal Ops**: These ops are used during transforms and should not be created by
   anyone except the internal passes to RD.

## Execution ##

This section describes how the computation executes after code-generation.
Although there are multiple execution models planned, only one (the iterator-
based model) has been implemented.

### Itereator-based execution ###

The iteration-based model centers around: (1) the creation of a stateful iterator
via the `rd.make_iterator` op, (2) repeatedly calling `rd.iterator_next` to retrieve
values.

#### Parallel Execution ####

When execution uses the `prefetched` operation, all operations above the prefetched
are executed asynchronously in a pipelined fashion using background threads.

When execution uses an `interleave` or `parallel_map`, work executes in parallel
using background threads.

When coroutines execute, they schedule work on a shared threadpool. When background
work has completed, the background tasks return and await being re-scheduled in
the future.

The API required of the threadpool is simply:

```c
void schedule(void (thunk*)(void* data), void* data);
```

The implementation of RD relies on the use of atomics and avoids using locks / mutexes.
While this enables very efficient execution, clients must not call 2 rd operations on an
iterator concurrently.

#### Coroutine State ####

... TODO: describe me. ...

## Memory management ##

For maximum efficiency, no memory allocation or deallocation occurs during iteration over a
dataset. The data structures required during execution of the core pipeline (excluding the
implementation of the data types of the elements themselves) are computed at compile time
and must be allocated and provided by the caller.

<!-- TODO: Add support for dynamic prefetched buffer sizes? -->

## Future features ##

v0.1 requirements (aka my TODO list):

 - Lowering to LLVM.
 - Make generic over element types.
 - Parallel execution (pipeline & data-parallel).

### Roadmap ###

 - Batching / window functions.
 - Tracing / profiling.
 - Random numbers.
 - Checkpointing / restoring.
 - Index-based / random access execution.

## Acknowledgements ##

Influence for RD comes from myraid sources, including [`tf.data`](https://tensorflow.org),
Apache Spark, Reactive Streaming, Dryad, [Penguin](https://github.com/saeta/penguin),
the Swift standard library, Rust and C++, and beyond.
