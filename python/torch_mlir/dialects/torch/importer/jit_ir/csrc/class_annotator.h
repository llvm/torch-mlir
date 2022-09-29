//===- class_annotator.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// Utilities for annotating Torch `c10::ClassType`
//
// We cannot intrusively add metadata to the ClassType, so we instead
// keep a parallel data structure.
//
// An annotation injects extra knowledge about the program which is not
// otherwise deducible. Thus, it is important that all annotations have a safe
// "no extra knowledge" state.
//
// Annotations should not be thought of at the MLIR level. They should express
// information at the level of the user-observable program semantics independent
// of implementation.
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRJITIRIMPORTER_CSRC_CLASS_ANNOTATOR_H
#define TORCHMLIRJITIRIMPORTER_CSRC_CLASS_ANNOTATOR_H

#include <torch/csrc/jit/ir/ir.h>

namespace torch_mlir {

// An annotation on a class's attribute (corresponds to a c10::ClassAttribute).
struct AttributeAnnotation {
  // Whether external access to this attribute is allowed.
  // The default "no knowledge" state of the program is that all attributes
  // can be externally accessed.
  bool isExported = true;

  std::string toString(const std::string &name);
};

// An annotation of an argument of a method.
//
// Note that the "self" parameter is considered an explicit argument as well.
struct ArgAnnotation {
  // If not None, represents information known about the shape of the
  // argument (the argument must be a tensor).
  // Each entry represents the size of each dimension of a tensor with known
  // rank. `-1` represents an unknown size along that dimension.
  c10::optional<std::vector<int64_t>> shape;

  // If not None, represents information known about the dtype of the argument
  // (the argument must be a tensor).
  c10::optional<c10::ScalarType> dtype;

  // If true, means that the user code will treat this argument as if it
  // has value semantics (the argument must be a tensor).
  //
  // In particular, this means that use code:
  // - expects the argument will not be mutated
  // - expects that any mutation to the argument internal to the program will
  //   not be reflected externally.
  //
  // A value of `false` preserves the default Torch semantics and is a
  // safe default.
  //
  // TODO: Also add a "last use" / "dead" flag, which enables more powerful
  // optimizations like reusing the input buffer for scratch space.
  bool hasValueSemantics = false;

  std::string toString(int argIndex);
};

// An annotation on a class's method (corresponds to a torch::jit::Function).
struct MethodAnnotation {
  // Whether external calls to this method are allowed.
  // The default "no knowledge" state of the program is that all methods
  // can be externally called.
  bool isExported = true;

  // Optional is not strictly needed here, but it prevents an unreasonably
  // large printout of the default ArgAnnotation for every method.
  c10::optional<std::vector<ArgAnnotation>> argAnnotations;

  std::string toString(const std::string &name);
};

// Annotations on a c10::ClassType.
//
// A c10::ClassType consists of attributes and methods, which are stored in
// arrays (the array elements know their names, but the storage is not keyed on
// the name). For each, we have an array of annotations that parallels the
// corresonding array (of either attributes or methods) held on the
// c10::ClassType.
//
// Note that c10::ClassType is in principle mutable, which can cause
// this data structure to get out of sync with it (this would be a problem with
// parallel arrays or string-keyed data structures). However, in practice the
// types tend to not change after being created from TorchScript.
//
// We make some mild efforts to check for mutation to the underlying, but
// they don't provide firm guarantees. Caveat emptor.
//
// Note: We do take advantage of this to assume that our annotation vectors
// don't resize (no invalidation of iterators).
class ClassAnnotation {
public:
  ClassAnnotation(c10::ClassTypePtr classType);

  // Get the attribute annotations.
  // The length and order is the same as `classType->getAttributes()`.
  std::vector<AttributeAnnotation> &getAttributeAnnotations();
  // Get the method annotations.
  // The length and order is the same as `classType->methods()`.
  std::vector<MethodAnnotation> &getMethodAnnotations();

  std::string toString();

private:
  // The c10::ClassType that we are annotating.
  //
  // Use a shared ptr type to keep the `ClassType *` alive.
  // We use a raw ptr as the map key where this class is the map value.
  c10::ClassTypePtr classType;
  std::vector<AttributeAnnotation> attributeAnnotations;
  std::vector<MethodAnnotation> methodAnnotations;
};

// A map of annotations on `c10::ClassType` names
using ClassAnnotationMap =
    std::map<std::string, std::unique_ptr<ClassAnnotation>>;

// A collection of class annotations + methods to create the annotations.
//
// This object is bound into Python, but the UI is quite poor. We expect
// some amount of Python metaprogramming syntax sugar to make it usable.
class ClassAnnotator {
public:
  ClassAnnotator() = default;
  // Export the path `exportedPath`, where the root of the traversal
  // is at `rootClassType`.
  //
  // For example, if `exportedPath = ['a', 'b']`, then `rootClassType` should
  // have a submodule `a` and that submodule should have a method or attribute
  // `b`.
  void exportPath(c10::ClassType &rootClassType,
                  std::vector<std::string> exportedPath);
  // Mark everything as not-exported.
  //
  // This is kind of useless by itself, but together with `exportPath` allows
  // exporting a subset of known names out of a larger collection of unknown
  // names.
  void exportNone(c10::ClassType &rootClassType);

  // Annotate shapes and dtypes of the arguments of a method at path `path` from
  // `rootClassType`.
  //
  // `argAnnotations` should be a list of 3-tuples, with the first element
  // being a list/tuple of integer sizes, and the second being a torch datatype
  // object, such as `torch.float32`, `torch.int8`, etc., and the last being
  // a "has value semantics" boolean.
  // These will be put into an `ArgAnnotation` struct -- see there for
  // precise definitions of the promised semantics of each entry.
  void annotateArgs(c10::ClassType &rootClassType,
                    std::vector<std::string> path,
                    std::vector<ArgAnnotation> argAnnotations);

  // The annotations collected so far.
  const ClassAnnotationMap &getAnnotationMap();

  // Get the ClassAnnotation corresponding to `classType`.
  ClassAnnotation &getOrCreateClassAnnotation(c10::ClassType *classType);

  // Helper to find the MethodAnnotation corresponding to a
  // torch::jit::Function, or null if not found.
  //
  // Users could in principle scan all annotations to find this, but it's more
  // efficient to maintain the reverse mapping directly.
  MethodAnnotation *
  getMethodAnnotationForFunction(torch::jit::Function *function);

  std::string toString();

private:
  // Traverse `path` starting from `rootClassType` to find the ClassType
  // of a presumed nested submodule. Throw an error if there is no such
  // submodule.
  c10::ClassType *getClassAtPath(c10::ClassType *rootClassType,
                                 std::vector<std::string> path);
  ClassAnnotationMap classAnnotations;
  // Reverse mapping used to service getMethodAnnotationForFunction.
  std::unordered_map<torch::jit::Function *, MethodAnnotation *>
      functionToMethodMap;
};

} // namespace torch_mlir

#endif // TORCHMLIRJITIRIMPORTER_CSRC_CLASS_ANNOTATOR_H
