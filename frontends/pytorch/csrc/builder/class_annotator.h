//===- class_annotations.h --------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
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

#ifndef NPCOMP_FRONTENDS_PYTORCH_CSRC_CLASS_ANNOTATOR_H
#define NPCOMP_FRONTENDS_PYTORCH_CSRC_CLASS_ANNOTATOR_H

#include "../pybind.h"

namespace torch_mlir {

// An annotation on a class's attribute (corresponds to a c10::ClassAttribute).
struct AttributeAnnotation {
  // Whether external access to this attribute is allowed.
  // The default "no knowledge" state of the program is that all attributes
  // can be externally accessed.
  bool isExported = true;

  std::string toString(const std::string &name);
};

// An annotation on a class's method (corresponds to a torch::jit::Function).
struct MethodAnnotation {
  // Whether external calls to this method are allowed.
  // The default "no knowledge" state of the program is that all methods
  // can be externally called.
  bool isExported = true;

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

// A map of annotations on `c10::ClassType`s
using ClassAnnotationMap =
    std::unordered_map<c10::ClassType *, std::unique_ptr<ClassAnnotation>>;

// A collection of class annotations + methods to create the annotations.
class ClassAnnotator {
public:
  ClassAnnotator() = default;
  // Export the path `exportedPath`, where the root of the traversal
  // is at `rootClassType`.
  //
  // For example, if `exportedPath = ['a', 'b']`, then `rootClassType` should
  // have a submodule `a` and that submodule should have a method or attribute
  // `b`.
  void exportPath(std::vector<std::string> exportedPath,
                  c10::ClassType &rootClassType);
  // Mark everything as not-exported.
  //
  // This is kind of useless by itself, but together with `exportPath` allows
  // exporting a subset of known names out of a larger collection of unknown
  // names.
  void exportNone(c10::ClassType &rootClassType);

  // The annotations collected so far.
  const ClassAnnotationMap &getAnnotationMap();

  // Get the ClassAnnotation corresponding to `classType`.
  ClassAnnotation &getOrCreateClassAnnotation(c10::ClassType *classType);

  std::string toString();

private:
  ClassAnnotationMap classAnnotations;
};

void initClassAnnotatorBindings(py::module &m);

} // namespace torch_mlir

#endif // NPCOMP_FRONTENDS_PYTORCH_CSRC_CLASS_ANNOTATOR_H
