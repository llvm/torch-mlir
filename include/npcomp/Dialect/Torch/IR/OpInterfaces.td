//===- OpInterface.td --------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

#ifndef NPCOMP_DIALECT_ATEN_IR_TORCH_OP_INTERFACES
#define NPCOMP_DIALECT_ATEN_IR_TORCH_OP_INTERFACES

def TorchKernelOpInterface : OpInterface<"TorchKernelOpInterface"> {
  let description = [{
    This interface provides metadata about ops that are isomorphic with a
    Torch kernel call. It is implemented by both the generic torch.kernel_call
    op and any recognized ops (i.e. in the ATen and other dialects) which
    are transformed from it and remain isomorphic with a Torch kernel (although
    typically with structural transformations that make them more amenable to
    compilation).

    As a reflection of the metadata that typically exists in the Torch op
    registry (type signatures, defaults, optionality, aliasing, op name, etc),
    it is expected that any op which implements this interface can be coerced
    heuristically back to a call into Torch itself as needed.

    Another purpose to carry this metadata is to facilitate conversions between
    ops at this level of abstraction. Since not all information about the
    original types and signatures is present in the ODS or MLIR types, keeping
    it accessible makes it possible to re-assemble a complete picture.
  }];

  let methods = [
    InterfaceMethod<"Metadata for the kernel",
      "::mlir::NPCOMP::Torch::KernelMetadata", "getTorchKernelMetadata">,
  ];
}

def TorchBuildableKernelOpInterface :
    OpInterface<"TorchBuildableKernelOpInterface"> {
  let description = [{
    Implemented by specific ops that can be constructed from a generic
    torch.kernel_call op via standard conversion heuristics. When taken in
    combination with the source op's TorchKernelOpInterface metedata, this
    provides the additional metadata to drive the conversion to a specific
    recognized op.

    The reason that this exists is that recognized ops (as opposed to the
    generic torch.kernel_call op) typically are formulated as befits the
    compiler, electing to leverage several systematic transformations in the
    conversion process from generic to the more structured forms:
      - Differentiate between tensors and arrays when mutability semantics
        are known. This allows most recognized ops to operate purely on
        values, leaving operation on references to those ops/operands/results
        that carry that semantic.
      - A single Torch kernel may correspond to several recognized ops that
        need to differentiate themselves based on signatures or other
        constraints that are relevant to the compiler.
      - Multiple Torch kernels or signatures may collapse down to a single
        recognized op if the transformation is systematic/pervasive and can
        be described by metadata. This is the case in at least the following
        cases:
          * Inplace kernels (those ending in trailing underscores) can expand
            to the non-inplace variant of the op with additional ops for
            loads and stores inserted as needed.
          * "Outref" signatures (many/most ops have an alternate signature that
            differs in the last argument, taking a Tensor to write the result
            into). These forms can be collapsed into the non-outref form by
            adding a trailing store.
          * Scalar promotion. There are often dedicated variants of kernels that
            have some of their parameters represented as scalars (for runtime
            efficiency, presumably). There is little benefit to the compiler
            to differentiating these forms: additional casts can be added to
            normalize to a tensor form, and any optimizations that result from
            the lower dimensionality should happen automatically.
      - There may be cosmetic type tweaks that simplify the representation. For
        example, a recognized op may elect to accept its int[] arguments as
        a tensor instead. Casts can be inserted systematically to bridge this
        when the metadata is known.

    Ops implementing this interface, in effect, encode the structure of the
    generic kernels that they can convert from, along with conversion hints. All
    such implementations of the interface for a kernel can be considered and the
    most salient chosen.

    By implementing this interface, it is assumed that there are no required
    attributes on the op, and constructing the operands and results according
    to the metadata will result in a valid operation.
  }];

  let methods = [
    StaticInterfaceMethod<"Gets kernel build metadata",
      "const ::mlir::NPCOMP::Torch::BuildKernelMetadata &",
      "getTorchBuildKernelMetadata">,
  ];
}
#endif // NPCOMP_DIALECT_ATEN_IR_TORCH_OP_INTERFACES
