//===- class_annotator.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "class_annotator.h"

#include <sstream>
#include <stdexcept>

using namespace torch_mlir;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Prefix every line of `s` with `linePrefix`.
static std::string indentString(const std::string &linePrefix,
                                const std::string &s) {
  std::stringstream is(s);
  std::stringstream os;
  std::string line;
  while (std::getline(is, line)) {
    os << linePrefix << line << "\n";
  }
  return os.str();
}

//===----------------------------------------------------------------------===//
// ClassAnnotation
//===----------------------------------------------------------------------===//

ClassAnnotation::ClassAnnotation(c10::ClassTypePtr classType)
    : classType(classType) {
  attributeAnnotations.resize(classType->getAttributes().size());
  methodAnnotations.resize(classType->methods().size());
}

std::vector<AttributeAnnotation> &ClassAnnotation::getAttributeAnnotations() {
  // Halfhearted attempt to ensure consistency if the class type has
  // been mutated.
  //
  // We can't easily guard against attributes being removed and
  // then other attributes being added, or types changed, etc. without
  // effectively mirroring the entire ClassType.
  assert(attributeAnnotations.size() == classType->getAttributes().size() &&
         "annotations out of sync. class has been mutated");

  return attributeAnnotations;
}

std::vector<MethodAnnotation> &ClassAnnotation::getMethodAnnotations() {
  // Halfhearted attempt to ensure consistency if the class type has
  // been mutated.
  //
  // We can't easily guard against methods being removed, added, or changed.
  assert(methodAnnotations.size() == classType->methods().size() &&
         "annotations out of sync. class has been mutated");

  return methodAnnotations;
}

//===----------------------------------------------------------------------===//
// ClassAnnotator
//===----------------------------------------------------------------------===//

static void exportNoneRecurse(ClassAnnotator &classAnnotator,
                              c10::ClassType *classType) {
  ClassAnnotation &classAnnotation =
      classAnnotator.getOrCreateClassAnnotation(classType);
  for (auto &attributeAnnotation : classAnnotation.getAttributeAnnotations()) {
    attributeAnnotation.isExported = false;
  }
  for (auto &methodAnnotation : classAnnotation.getMethodAnnotations()) {
    methodAnnotation.isExported = false;
  }
  for (auto &classAttribute : classType->getAttributes()) {
    if (auto childClassType =
            classAttribute.getType()->cast<c10::ClassType>()) {
      exportNoneRecurse(classAnnotator, childClassType.get());
    }
  }
}

void ClassAnnotator::exportNone(c10::ClassType &rootClassType) {
  exportNoneRecurse(*this, &rootClassType);
}

void ClassAnnotator::exportPath(c10::ClassType &rootClassType,
                                std::vector<std::string> exportedPath) {
  if (exportedPath.size() == 0) {
    throw std::invalid_argument(
        "Empty exported path. Can only export a property of a class.");
  }
  c10::ClassType *classType =
      getClassAtPath(&rootClassType, c10::ArrayRef<std::string>(exportedPath)
                                         .slice(0, exportedPath.size() - 1)
                                         .vec());

  if (!classType->findAttribute(exportedPath.back()) &&
      !classType->findMethod(exportedPath.back())) {
    std::stringstream ss;
    ss << "class '" << classType->name()->qualifiedName()
       << "' does not have a method or attribute called '"
       << exportedPath.back() << "'";
    throw std::invalid_argument(ss.str());
  }
  ClassAnnotation &classAnnotation = getOrCreateClassAnnotation(classType);
  std::vector<AttributeAnnotation> &attributeAnnotations =
      classAnnotation.getAttributeAnnotations();
  const std::vector<c10::ClassAttribute> &classAttributes =
      classType->getAttributes();
  for (int i = 0, e = classAttributes.size(); i != e; i++) {
    if (classAttributes[i].getName() == exportedPath.back()) {
      attributeAnnotations[i].isExported = true;
    }
  }

  std::vector<MethodAnnotation> &methodAnnotations =
      classAnnotation.getMethodAnnotations();
  const std::vector<torch::jit::Function *> &methods = classType->methods();
  for (int i = 0, e = methods.size(); i != e; i++) {
    if (methods[i]->name() == exportedPath.back()) {
      methodAnnotations[i].isExported = true;
    }
  }
}

const ClassAnnotationMap &ClassAnnotator::getAnnotationMap() {
  return classAnnotations;
}

ClassAnnotation &
ClassAnnotator::getOrCreateClassAnnotation(c10::ClassType *classType) {
  auto className = classType->name()->qualifiedName();
  auto it = classAnnotations.find(className);
  if (it == classAnnotations.end()) {
    auto newAnnotation = std::make_unique<ClassAnnotation>(
        classType->shared_from_this()->cast<c10::ClassType>());
    it = classAnnotations.insert({className, std::move(newAnnotation)}).first;
    for (int i = 0, e = classType->methods().size(); i != e; i++) {
      functionToMethodMap[classType->methods()[i]] =
          &it->second->getMethodAnnotations()[i];
    }
  }
  return *it->second;
}

static void fillArgAnnotations(MethodAnnotation &methodAnnotation,
                               const std::vector<ArgAnnotation> &argAnnotations,
                               torch::jit::Function *function) {
  if (argAnnotations.size() != function->num_inputs()) {

    std::ostringstream oss;
    oss << "There must be one argument annotation per function parameter. "
        << "Including 'self' the number of argument annotations is: "
        << argAnnotations.size()
        << ". The number of function parameters is: " << function->num_inputs()
        << ". ";
    const auto &args = function->getSchema().arguments();
    if (args.size() > 0) {
      oss << "The function signature is (";
      oss << args[0];
      for (auto iter = args.begin() + 1; iter != args.end(); iter++) {
        oss << ", " << *iter;
      }
      oss << ')' << '.';
    }
    throw std::invalid_argument(oss.str());
  }
  if (!methodAnnotation.argAnnotations.has_value()) {
    methodAnnotation.argAnnotations.emplace(function->num_inputs(),
                                            ArgAnnotation{});
  }

  methodAnnotation.argAnnotations = argAnnotations;
}

void ClassAnnotator::annotateArgs(c10::ClassType &rootClassType,
                                  std::vector<std::string> path,
                                  std::vector<ArgAnnotation> argAnnotations) {
  if (path.size() == 0) {
    throw std::invalid_argument("Empty annotated path. Can only annotate "
                                "shapes/dtypes of a method of a class.");
  }
  c10::ClassType *classType = getClassAtPath(
      &rootClassType,
      c10::ArrayRef<std::string>(path).slice(0, path.size() - 1).vec());

  // Throw error if no method on the class of the specified name.
  torch::jit::Function *function = &classType->getMethod(path.back());

  ClassAnnotation &classAnnotation = getOrCreateClassAnnotation(classType);
  std::vector<MethodAnnotation> &methodAnnotations =
      classAnnotation.getMethodAnnotations();
  const std::vector<torch::jit::Function *> &methods = classType->methods();
  for (int i = 0, e = methods.size(); i != e; i++) {
    if (methods[i]->name() == path.back()) {
      fillArgAnnotations(methodAnnotations[i], argAnnotations, function);
    }
  }

  return;
}

c10::ClassType *ClassAnnotator::getClassAtPath(c10::ClassType *rootClassType,
                                               std::vector<std::string> path) {
  c10::ClassType *classType = rootClassType;
  // Reverse so that pop_back gives us the initial atoms first.
  std::reverse(path.begin(), path.end());
  while (!path.empty()) {
    // This will throw in case of missing attribute.
    c10::TypePtr childType = classType->getAttribute(path.back());
    c10::ClassTypePtr childClassType = childType->cast<c10::ClassType>();
    if (!childClassType) {
      std::stringstream ss;
      ss << "class '" << classType->name()->qualifiedName()
         << "' does not have a submodule in attribute '" << path.back() << "'";
      throw std::invalid_argument(ss.str());
    }
    path.pop_back();
    classType = childClassType.get();
  }
  return classType;
}

//===----------------------------------------------------------------------===//
// Helper methods
//===----------------------------------------------------------------------===//
MethodAnnotation *
ClassAnnotator::getMethodAnnotationForFunction(torch::jit::Function *function) {
  auto it = functionToMethodMap.find(function);
  if (it == functionToMethodMap.end()) {
    return nullptr;
  }
  return it->second;
}

//===----------------------------------------------------------------------===//
// toString methods
//===----------------------------------------------------------------------===//

std::string AttributeAnnotation::toString(const std::string &name) {
  std::stringstream ss;
  ss << "AttributeAnnotation('" << name << "') {\n";
  ss << "  isExported = " << (isExported ? "true" : "false") << "\n";
  ss << "}\n";
  return ss.str();
}

std::string ArgAnnotation::toString(int argIndex) {
  std::stringstream ss;
  ss << "ArgAnnotation(" << argIndex << ") {\n";
  ss << "  dtype = " << (dtype ? c10::toString(*dtype) : "<none>") << "\n";
  ss << "  shape = ";
  if (shape) {
    ss << "[";
    for (int i = 0, e = shape.value().size(); i != e; i++) {
      if (i) {
        ss << ", ";
      }
      ss << shape.value()[i];
    }
    ss << "]\n";
  } else {
    ss << "<none>\n";
  }
  ss << "  hasValueSemantics = " << (hasValueSemantics ? "true" : "false")
     << "\n";
  ss << "}\n";
  return ss.str();
}

std::string MethodAnnotation::toString(const std::string &name) {
  std::stringstream ss;
  ss << "MethodAnnotation('" << name << "') {\n";
  ss << "  isExported = " << (isExported ? "true" : "false") << "\n";
  ss << "  argAnnotations =";
  if (argAnnotations) {
    ss << "\n";
    for (int i = 0, e = argAnnotations.value().size(); i < e; i++) {
      ss << indentString("    ", argAnnotations.value()[i].toString(i));
    }
  } else {
    ss << " <none>\n";
  }
  ss << "}\n";
  return ss.str();
}

std::string ClassAnnotation::toString() {
  std::stringstream ss;
  ss << "ClassAnnotation('" << classType->name()->qualifiedName() << "') {\n";

  const std::vector<c10::ClassAttribute> &classAttributes =
      classType->getAttributes();
  for (int i = 0, e = classAttributes.size(); i != e; i++) {
    ss << indentString(
        "  ", attributeAnnotations[i].toString(classAttributes[i].getName()));
  }
  const std::vector<torch::jit::Function *> &methods = classType->methods();
  for (int i = 0, e = methods.size(); i != e; i++) {
    ss << indentString("  ", methodAnnotations[i].toString(methods[i]->name()));
  }
  ss << "}\n";
  return ss.str();
}

std::string ClassAnnotator::toString() {
  std::stringstream ss;
  ss << "ClassAnnotator {\n";
  for (auto &p : classAnnotations) {
    ss << indentString("  ", p.second->toString());
  }
  ss << "}\n";
  return ss.str();
}
