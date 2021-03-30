//===- class_annotator.cpp ------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "class_annotator.h"

#include <stdexcept>

#include "torch/csrc/Dtype.h"

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

std::vector<AttributeAnnotation> &
ClassAnnotation::getAttributeAnnotations() {
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

std::vector<MethodAnnotation> &
ClassAnnotation::getMethodAnnotations() {
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
    if (auto childClassType = classAttribute.getType()->cast<c10::ClassType>()) {
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
  auto it = classAnnotations.find(classType);
  if (it == classAnnotations.end()) {
    auto newAnnotation = std::make_unique<ClassAnnotation>(
        classType->shared_from_this()->cast<c10::ClassType>());
    it = classAnnotations.insert({classType, std::move(newAnnotation)}).first;
    for (int i = 0, e = classType->methods().size(); i != e; i++) {
      functionToMethodMap[classType->methods()[i]] =
          &it->second->getMethodAnnotations()[i];
    }
  }
  return *it->second;
}

static c10::ScalarType convertToC10ScalarType(py::object obj) {
  if (THPDtype_Check(obj.ptr())) {
    // Need reinterpret_cast, since no C++-level inheritance is involved.
    THPDtype *dtype = reinterpret_cast<THPDtype *>(obj.ptr());
    return dtype->scalar_type;
  }
  std::stringstream ss;
  ss << "unsupported scalar type '" << obj << "'";
  throw std::invalid_argument(ss.str());
}

static void fillArgAnnotations(MethodAnnotation &methodAnnotation,
                               py::list pyArgAnnotations,
                               torch::jit::Function *function) {
  if (pyArgAnnotations.size() != function->num_inputs()) {
    throw std::invalid_argument("Arg annotations should have one entry per "
                                "function parameter (including self).");
  }
  if (!methodAnnotation.argAnnotations.has_value()) {
    methodAnnotation.argAnnotations.emplace(function->num_inputs(),
                                            ArgAnnotation{});
  }
  std::vector<ArgAnnotation> &argAnnotations =
      methodAnnotation.argAnnotations.value();
  for (int i = 0, e = argAnnotations.size(); i != e; i++) {
    if (pyArgAnnotations[i].is_none()) {
      continue;
    }
    auto tuple = py::cast<py::tuple>(pyArgAnnotations[i]);
    auto shape = tuple[0];
    auto dtype = tuple[1];
    if (!shape.is_none()) {
      argAnnotations[i].shape = py::cast<std::vector<int64_t>>(shape);
    }
    if (!dtype.is_none()) {
      argAnnotations[i].dtype = convertToC10ScalarType(dtype);
    }
  };
}

void ClassAnnotator::annotateShapesAndDtypes(c10::ClassType &rootClassType,
                                             std::vector<std::string> path,
                                             py::list argAnnotations) {
  if (path.size() == 0) {
    throw std::invalid_argument("Empty annotated path. Can only annotate "
                                "shapes/dtypes of a method of a class.");
  }
  c10::ClassType *classType =
      getClassAtPath(&rootClassType, c10::ArrayRef<std::string>(path)
                                         .slice(0, path.size() - 1)
                                         .vec());

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

void torch_mlir::initClassAnnotatorBindings(py::module &m) {
  py::class_<ClassAnnotator>(m, "ClassAnnotator")
      .def(py::init<>())
      .def("exportPath", &ClassAnnotator::exportPath)
      .def("exportNone", &ClassAnnotator::exportNone)
      .def("annotateShapesAndDtypes", &ClassAnnotator::annotateShapesAndDtypes)
      .def("__repr__", &ClassAnnotator::toString);
}
