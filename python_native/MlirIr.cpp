//===- MlirIr.cpp - MLIR IR Bindings --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MlirIr.h"
#include "NpcompModule.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Forward declarations
//===----------------------------------------------------------------------===//
struct PyContext;

/// Parses an MLIR module from a string.
/// For maximum efficiency, the `contents` should be zero terminated.
static OwningModuleRef parseMLIRModuleFromString(StringRef contents,
                                                 MLIRContext *context);

//===----------------------------------------------------------------------===//
// Internal only template definitions
// Since it is only legal to use explicit instantiations of templates in
// mlir_ir.h, implementations are kept in this module to keep things scoped
// well for the compiler.
//===----------------------------------------------------------------------===//

template <typename ListTy, typename ItemWrapperTy>
void PyIpListWrapper<ListTy, ItemWrapperTy>::bind(py::module m,
                                                  const char *className) {
  struct PyItemIterator : public llvm::iterator_adaptor_base<
                              PyItemIterator, typename ListTy::iterator,
                              typename std::iterator_traits<
                                  typename ListTy::iterator>::iterator_category,
                              typename ListTy::value_type> {
    PyItemIterator() = default;
    PyItemIterator(typename ListTy::iterator &&other)
        : PyItemIterator::iterator_adaptor_base(std::move(other)) {}
    ItemWrapperTy operator*() const { return ItemWrapperTy(*this->I); }
  };

  py::class_<ThisTy>(m, className)
      .def_property_readonly(
          "front",
          [](ThisTy &self) { return ItemWrapperTy(self.list.front()); })
      .def("__len__", [](ThisTy &self) { return self.list.size(); })
      .def("__iter__",
           [](ThisTy &self) {
             PyItemIterator begin(self.list.begin());
             PyItemIterator end(self.list.end());
             return py::make_iterator(begin, end);
           },
           py::keep_alive<0, 1>());
}

//===----------------------------------------------------------------------===//
// Explicit template instantiations
//===----------------------------------------------------------------------===//

template class PyIpListWrapper<Region::BlockListType, PyBlockRef>;
using PyBlockList = PyIpListWrapper<Region::BlockListType, PyBlockRef>;

template class PyIpListWrapper<Block::OpListType, PyOperationRef>;
using PyOperationList = PyIpListWrapper<Block::OpListType, PyOperationRef>;

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

Type mapBufferFormatToType(MLIRContext *context, const std::string &format,
                           py::ssize_t itemSize) {
  // Floating point formats.
  if (format == "f")
    return FloatType::getF32(context);
  if (format == "d")
    return FloatType::getF64(context);
  if (format == "D")
    return ComplexType::get(FloatType::getF64(context));

  // Signed integer formats.
  if (format == "b" || format == "h" || format == "i" || format == "l" ||
      format == "L") {
    unsigned width = itemSize * 8;
    return IntegerType::get(width, IntegerType::SignednessSemantics::Signed,
                            context);
  }

  // Unsigned integer format.
  if (format == "B" || format == "H" || format == "I" || format == "k" ||
      format == "K") {
    unsigned width = itemSize * 8;
    return IntegerType::get(width, IntegerType::SignednessSemantics::Unsigned,
                            context);
  }

  return Type();
}

/// Creates a DenseElementsAttr from a python buffer which must have been
/// requested to be C-Contiguous.
Attribute createDenseElementsAttrFromBuffer(MLIRContext *context,
                                            py::buffer_info &array) {
  Type elementType =
      mapBufferFormatToType(context, array.format, array.itemsize);
  if (!elementType) {
    throw py::raiseValueError(
        "Unsupported buffer/array type for conversion to DenseElementsAttr");
  }

  SmallVector<int64_t, 4> shape(array.shape.begin(),
                                array.shape.begin() + array.ndim);
  RankedTensorType type = RankedTensorType::get(shape, elementType);
  const char *rawBufferPtr = reinterpret_cast<const char *>(array.ptr);
  ArrayRef<char> rawBuffer(rawBufferPtr, array.size * array.itemsize);
  return DenseElementsAttr::getFromRawBuffer(type, rawBuffer, false);
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

/// RAII class to capture diagnostics for later reporting back to the python
/// layer.
class DiagnosticCapture {
public:
  DiagnosticCapture(mlir::MLIRContext *mlir_context)
      : mlir_context(mlir_context) {
    handler_id = mlir_context->getDiagEngine().registerHandler(
        [&](Diagnostic &d) -> LogicalResult {
          diagnostics.push_back(std::move(d));
          return success();
        });
  }

  ~DiagnosticCapture() {
    if (mlir_context) {
      mlir_context->getDiagEngine().eraseHandler(handler_id);
    }
  }
  DiagnosticCapture(DiagnosticCapture &&other) {
    mlir_context = other.mlir_context;
    diagnostics.swap(other.diagnostics);
    handler_id = other.handler_id;
    other.mlir_context = nullptr;
  }

  std::vector<mlir::Diagnostic> &getDiagnostics() { return diagnostics; }

  // Consumes/clears diagnostics.
  std::string consumeDiagnosticsAsString(const char *error_message);
  void clearDiagnostics() { diagnostics.clear(); }

private:
  MLIRContext *mlir_context;
  std::vector<mlir::Diagnostic> diagnostics;
  mlir::DiagnosticEngine::HandlerID handler_id;
};

//===----------------------------------------------------------------------===//
// PyDialectHelper
//===----------------------------------------------------------------------===//

void PyDialectHelper::bind(py::module m) {
  py::class_<PyDialectHelper>(m, "DialectHelper")
      .def(py::init<std::shared_ptr<PyContext>>())
      .def_property_readonly("builder",
                             [](PyDialectHelper &self) -> PyBaseOpBuilder & {
                               return self.pyOpBuilder;
                             })
      .def_property_readonly(
          "context",
          [](PyDialectHelper &self) -> std::shared_ptr<PyContext> {
            return self.context;
          })
      .def("op",
           [](PyDialectHelper &self, const std::string &opNameStr,
              std::vector<PyType> pyResultTypes,
              std::vector<PyValue> pyOperands,
              llvm::Optional<PyAttribute> attrs) -> PyOperationRef {
             OpBuilder &opBuilder = self.pyOpBuilder.getBuilder(false);
             Location loc = UnknownLoc::get(opBuilder.getContext());
             OperationName opName(opNameStr, opBuilder.getContext());
             SmallVector<Type, 4> types(pyResultTypes.begin(),
                                        pyResultTypes.end());
             SmallVector<Value, 4> operands(pyOperands.begin(),
                                            pyOperands.end());
             MutableDictionaryAttr attrList;
             if (attrs) {
               auto dictAttrs = attrs->attr.dyn_cast<DictionaryAttr>();
               if (!dictAttrs) {
                 throw py::raiseValueError(
                     "Expected `attrs` to be a DictionaryAttr");
               }
               attrList = MutableDictionaryAttr(dictAttrs);
             }
             Operation *op =
                 Operation::create(loc, opName, types, operands, attrList);
             opBuilder.insert(op);
             return op;
           },
           py::arg("op_name"), py::arg("result_types"), py::arg("operands"),
           py::arg("attrs") = llvm::Optional<PyAttribute>())
      .def("func_op",
           [](PyDialectHelper &self, const std::string &name, PyType type,
              bool createEntryBlock) {
             auto functionType = type.type.dyn_cast_or_null<FunctionType>();
             if (!functionType) {
               throw py::raiseValueError("Illegal function type");
             }
             OpBuilder &opBuilder = self.pyOpBuilder.getBuilder(true);
             Location loc = UnknownLoc::get(opBuilder.getContext());
             // TODO: Add function and arg/result attributes.
             FuncOp op =
                 opBuilder.create<FuncOp>(loc, StringRef(name), functionType,
                                          /*attrs=*/ArrayRef<NamedAttribute>());
             if (createEntryBlock) {
               Block *entryBlock = new Block();
               entryBlock->addArguments(functionType.getInputs());
               op.getBody().push_back(entryBlock);
               opBuilder.setInsertionPointToStart(entryBlock);
             }
             return PyOperationRef(op);
           },
           py::arg("name"), py::arg("type"),
           py::arg("create_entry_block") = false,
           R"(Creates a new `func` op, optionally creating an entry block.
              If an entry block is created, the builder will be positioned
              to its start.)")
      .def("return_op",
           [](PyDialectHelper &self, std::vector<PyValue> pyOperands) {
             OpBuilder &opBuilder = self.pyOpBuilder.getBuilder(true);
             Location loc = UnknownLoc::get(opBuilder.getContext());
             SmallVector<Value, 4> operands(pyOperands.begin(),
                                            pyOperands.end());
             return PyOperationRef(opBuilder.create<ReturnOp>(loc, operands));
           })
      .def("constant_op",
           [](PyDialectHelper &self, PyType type, PyAttribute value) {
             OpBuilder &opBuilder = self.pyOpBuilder.getBuilder(true);
             Location loc = UnknownLoc::get(opBuilder.getContext());
             return PyOperationRef(
                 opBuilder.create<ConstantOp>(loc, type.type, value.attr));
           })

      // Types.
      .def_property_readonly("index_type",
                             [](PyDialectHelper &self) -> PyType {
                               return IndexType::get(&self.context->context);
                             })
      .def("integer_type",
           [](PyDialectHelper &self, unsigned width) -> PyType {
             return IntegerType::get(width, &self.context->context);
           },
           py::arg("width") = 32)
      .def_property_readonly("i1_type",
                             [](PyDialectHelper &self) -> PyType {
                               return IntegerType::get(1,
                                                       &self.context->context);
                             })
      .def_property_readonly("i16_type",
                             [](PyDialectHelper &self) -> PyType {
                               return IntegerType::get(32,
                                                       &self.context->context);
                             })
      .def_property_readonly("i32_type",
                             [](PyDialectHelper &self) -> PyType {
                               return IntegerType::get(32,
                                                       &self.context->context);
                             })
      .def_property_readonly("i64_type",
                             [](PyDialectHelper &self) -> PyType {
                               return IntegerType::get(64,
                                                       &self.context->context);
                             })
      .def_property_readonly("f32_type",
                             [](PyDialectHelper &self) -> PyType {
                               return FloatType::get(StandardTypes::F32,
                                                     &self.context->context);
                             })
      .def_property_readonly("f64_type",
                             [](PyDialectHelper &self) -> PyType {
                               return FloatType::get(StandardTypes::F64,
                                                     &self.context->context);
                             })
      .def("tensor_type",
           [](PyDialectHelper &self, PyType elementType,
              llvm::Optional<std::vector<int64_t>> shape) -> PyType {
             if (!elementType.type) {
               throw py::raiseValueError("Null element type");
             }
             if (shape) {
               return RankedTensorType::get(*shape, elementType.type);
             } else {
               return UnrankedTensorType::get(elementType.type);
             }
           },
           py::arg("element_type"),
           py::arg("shape") = llvm::Optional<std::vector<int64_t>>())
      .def("function_type",
           [](PyDialectHelper &self, std::vector<PyType> inputs,
              std::vector<PyType> results) -> PyType {
             llvm::SmallVector<Type, 4> inputTypes;
             llvm::SmallVector<Type, 1> resultTypes;
             for (auto input : inputs) {
               inputTypes.push_back(input.type);
             }
             for (auto result : results) {
               resultTypes.push_back(result.type);
             }
             return FunctionType::get(inputTypes, resultTypes,
                                      &self.context->context);
           });
}

//===----------------------------------------------------------------------===//
// Module initialization
//===----------------------------------------------------------------------===//

void defineMlirIrModule(py::module m) {
  m.doc() = "Python bindings for constructs in the mlir/IR library";

  // Python only types.
  PyDialectHelper::bind(m);

  // Utility types.
  PyBlockList::bind(m, "BlockList");
  PyOperationList::bind(m, "OperationList");

  // Wrapper types.
  PyAttribute::bind(m);
  PyBaseOperation::bind(m);
  PyBaseOpBuilder::bind(m);
  PyBlockRef::bind(m);
  PyContext::bind(m);
  PyModuleOp::bind(m);
  PyOperationRef::bind(m);
  PyOpBuilder::bind(m);
  PyRegionRef::bind(m);
  PySymbolTable::bind(m);
  PyType::bind(m);
  PyValue::bind(m);
}

//===----------------------------------------------------------------------===//
// PyContext
//===----------------------------------------------------------------------===//

void PyContext::bind(py::module m) {
  py::class_<PyContext, std::shared_ptr<PyContext>>(m, "MLIRContext")
      .def(py::init<>([]() {
        // Need explicit make_shared to avoid UB with enable_shared_from_this.
        return std::make_shared<PyContext>();
      }))
      .def("new_module",
           [&](PyContext &self) -> PyModuleOp {
             Location loc = UnknownLoc::get(&self.context);
             auto m = ModuleOp::create(loc);
             return PyModuleOp(self.shared_from_this(), m);
           })
      .def("parse_asm", &PyContext::parseAsm)
      .def("new_builder",
           [](PyContext &self) {
             // Note: we collapse the Builder and OpBuilder into one because
             // there is little reason to expose the inheritance hierarchy to
             // Python.
             return PyOpBuilder(self);
           })
      // Salient functions from Builder.
      .def("parse_type",
           [](PyContext &self, const std::string &asmText) {
             Type t = parseType(asmText, &self.context);
             if (!t) {
               std::string message = "Unable to parse MLIR type: ";
               message.append(asmText);
               throw py::raiseValueError(message);
             }
             return PyType(t);
           })
      .def("index_attr",
           [](PyContext &self, int64_t indexValue) -> PyAttribute {
             return IntegerAttr::get(IndexType::get(&self.context), indexValue);
           })
      .def("string_attr",
           [](PyContext &self, const std::string &s) -> PyAttribute {
             return StringAttr::get(s, &self.context);
           })
      .def("bytes_attr",
           [](PyContext &self, py::bytes bytes) -> PyAttribute {
             char *buffer;
             ssize_t length;
             if (PYBIND11_BYTES_AS_STRING_AND_SIZE(bytes.ptr(), &buffer,
                                                   &length)) {
               throw py::raiseValueError("Cannot extract bytes");
             }
             return StringAttr::get(StringRef(buffer, length), &self.context);
           })
      .def("flat_symbol_ref_attr",
           [](PyContext &self, const std::string &s) -> PyAttribute {
             return FlatSymbolRefAttr::get(s, &self.context);
           })
      .def("dictionary_attr",
           [](PyContext &self, py::dict d) -> PyAttribute {
             SmallVector<NamedAttribute, 4> attrs;
             for (auto &it : d) {
               auto key = it.first.cast<std::string>();
               auto value = it.second.cast<PyAttribute>();
               auto keyIdent = Identifier::get(key, &self.context);
               attrs.emplace_back(keyIdent, value.attr);
             }
             return DictionaryAttr::get(attrs, &self.context);
           })
      .def("dense_elements_attr",
           [](PyContext &self, py::buffer array) -> PyAttribute {
             // Request a contiguous view.
             int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
             Py_buffer *view = new Py_buffer();
             if (PyObject_GetBuffer(array.ptr(), view, flags) != 0) {
               delete view;
               throw py::error_already_set();
             }
             py::buffer_info array_info(view);
             return createDenseElementsAttrFromBuffer(&self.context,
                                                      array_info);
           },
           py::arg("array"));
}

PyModuleOp PyContext::parseAsm(const std::string &asm_text) {
  // Arrange to get a view that includes a terminating null to avoid
  // additional copy.
  // TODO: Consider using the buffer protocol to access and avoid more copies.
  const char *asm_chars = asm_text.c_str();
  StringRef asm_sr(asm_chars, asm_text.size() + 1);

  // TODO: Output non failure diagnostics (somewhere)
  DiagnosticCapture diag_capture(&context);
  auto module_ref = parseMLIRModuleFromString(asm_sr, &context);
  if (!module_ref) {
    throw py::raiseValueError(
        diag_capture.consumeDiagnosticsAsString("Error parsing ASM"));
  }
  return PyModuleOp{shared_from_this(), module_ref.release()};
}

//===----------------------------------------------------------------------===//
// PyBaseOperation
//===----------------------------------------------------------------------===//

PyBaseOperation::~PyBaseOperation() = default;

void PyBaseOperation::bind(py::module m) {
  py::class_<PyBaseOperation>(m, "BaseOperation")
      .def_property_readonly(
          "name",
          [](PyBaseOperation &self) {
            return std::string(self.getOperation()->getName().getStringRef());
          })
      .def_property_readonly("is_registered",
                             [](PyBaseOperation &self) {
                               return self.getOperation()->isRegistered();
                             })
      .def_property_readonly("num_regions",
                             [](PyBaseOperation &self) {
                               return self.getOperation()->getNumRegions();
                             })
      .def_property_readonly("results",
                             [](PyBaseOperation &self) {
                               auto *op = self.getOperation();
                               std::vector<PyValue> results(op->result_begin(),
                                                            op->result_end());
                               return results;
                             })
      .def_property_readonly("result",
                             [](PyBaseOperation &self) -> PyValue {
                               auto *op = self.getOperation();
                               if (op->getNumResults() != 1) {
                                 throw py::raiseValueError(
                                     "Operation does not have 1 result");
                               }
                               return op->getOpResult(0);
                             })
      .def("region",
           [](PyBaseOperation &self, int index) {
             auto *op = self.getOperation();
             if (index < 0 || index >= op->getNumRegions()) {
               throw py::raisePyError(PyExc_IndexError,
                                      "Region index out of bounds");
             }
             return PyRegionRef(op->getRegion(index));
           })
      .def_property_readonly("first_block", [](PyBaseOperation &self) {
        Operation *op = self.getOperation();
        assert(op);
        if (op->getNumRegions() == 0) {
          throw py::raiseValueError("Op has no regions");
        }
        auto &region = op->getRegion(0);
        if (region.empty()) {
          throw py::raiseValueError("Op has no blocks");
        }
        return PyBlockRef(region.front());
      });
}

//===----------------------------------------------------------------------===//
// PyOperationRef
//===----------------------------------------------------------------------===//

PyOperationRef::~PyOperationRef() = default;
void PyOperationRef::bind(py::module m) {
  py::class_<PyOperationRef, PyBaseOperation>(m, "OperationRef");
}

Operation *PyOperationRef::getOperation() { return operation; }

//===----------------------------------------------------------------------===//
// PyModuleOp
//===----------------------------------------------------------------------===//

PyModuleOp::~PyModuleOp() = default;
void PyModuleOp::bind(py::module m) {
  py::class_<PyModuleOp, PyBaseOperation>(m, "ModuleOp")
      .def_property_readonly("context",
                             [](PyModuleOp &self) { return self.context; })
      .def("to_asm", &PyModuleOp::toAsm, py::arg("debug_info") = false,
           py::arg("pretty") = false, py::arg("large_element_limit") = -1);
}

Operation *PyModuleOp::getOperation() { return moduleOp; }

std::string PyModuleOp::toAsm(bool enableDebugInfo, bool prettyForm,
                              int64_t largeElementLimit) {
  // Print to asm.
  std::string asmOutput;
  llvm::raw_string_ostream sout(asmOutput);
  OpPrintingFlags printFlags;
  if (enableDebugInfo) {
    printFlags.enableDebugInfo(prettyForm);
  }
  if (largeElementLimit >= 0) {
    printFlags.elideLargeElementsAttrs(largeElementLimit);
  }
  moduleOp.print(sout, printFlags);
  return sout.str();
}

static OwningModuleRef parseMLIRModuleFromString(StringRef contents,
                                                 MLIRContext *context) {
  std::unique_ptr<llvm::MemoryBuffer> contents_buffer;
  if (contents.back() == 0) {
    // If it has a nul terminator, just use as-is.
    contents_buffer = llvm::MemoryBuffer::getMemBuffer(contents.drop_back());
  } else {
    // Otherwise, make a copy.
    contents_buffer = llvm::MemoryBuffer::getMemBufferCopy(contents, "EMBED");
  }

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(contents_buffer), llvm::SMLoc());
  OwningModuleRef mlir_module = parseSourceFile(source_mgr, context);
  return mlir_module;
}

// Custom location printer that prints prettier, multi-line file output
// suitable for human readable error messages. The standard printer just prints
// a long nested expression not particularly human friendly). Note that there
// is a location pretty printer in the MLIR AsmPrinter. It is private and
// doesn't do any path shortening, which seems to make long Python stack traces
// a bit easier to scan.
// TODO: Upstream this.
void printLocation(Location loc, raw_ostream &out) {
  switch (loc->getKind()) {
  case StandardAttributes::OpaqueLocation:
    printLocation(loc.cast<OpaqueLoc>().getFallbackLocation(), out);
    break;
  case StandardAttributes::UnknownLocation:
    out << "  [unknown location]\n";
    break;
  case StandardAttributes::FileLineColLocation: {
    auto line_col_loc = loc.cast<FileLineColLoc>();
    StringRef this_filename = line_col_loc.getFilename();
    auto slash_pos = this_filename.find_last_of("/\\");
    // We print both the basename and extended names with a structure like
    // `foo.py:35:4`. Even though technically the line/col
    // information is redundant to include in both names, having it on both
    // makes it easier to paste the paths into an editor and jump to the exact
    // location.
    std::string line_col_suffix = ":" + std::to_string(line_col_loc.getLine()) +
                                  ":" +
                                  std::to_string(line_col_loc.getColumn());
    bool has_basename = false;
    StringRef basename = this_filename;
    if (slash_pos != StringRef::npos) {
      has_basename = true;
      basename = this_filename.substr(slash_pos + 1);
    }
    out << "  at: " << basename << line_col_suffix;
    if (has_basename) {
      StringRef extended_name = this_filename;
      // Print out two tabs, as basenames usually vary in length by more than
      // one tab width.
      out << "\t\t( " << extended_name << line_col_suffix << " )";
    }
    out << "\n";
    break;
  }
  case StandardAttributes::NameLocation: {
    auto nameLoc = loc.cast<NameLoc>();
    out << "  @'" << nameLoc.getName() << "':\n";
    auto childLoc = nameLoc.getChildLoc();
    if (!childLoc.isa<UnknownLoc>()) {
      out << "(...\n";
      printLocation(childLoc, out);
      out << ")\n";
    }
    break;
  }
  case StandardAttributes::CallSiteLocation: {
    auto call_site = loc.cast<CallSiteLoc>();
    printLocation(call_site.getCaller(), out);
    printLocation(call_site.getCallee(), out);
    break;
  }
  }
}

//===----------------------------------------------------------------------===//
// PySymbolTable
//===----------------------------------------------------------------------===//

void PySymbolTable::bind(py::module m) {
  py::class_<PySymbolTable>(m, "SymbolTable")
      .def_property_readonly_static("symbol_attr_name",
                                    [](const py::object &) {
                                      auto sr =
                                          SymbolTable::getSymbolAttrName();
                                      return py::str(sr.data(), sr.size());
                                    })
      .def_property_readonly_static(
          "visibility_attr_name", [](const py::object &) {
            auto sr = SymbolTable::getVisibilityAttrName();
            return py::str(sr.data(), sr.size());
          });
}

//===----------------------------------------------------------------------===//
// DiagnosticCapture
//===----------------------------------------------------------------------===//

std::string
DiagnosticCapture::consumeDiagnosticsAsString(const char *error_message) {
  std::string s;
  llvm::raw_string_ostream sout(s);
  bool first = true;
  if (error_message) {
    sout << error_message;
    first = false;
  }
  for (auto &d : diagnostics) {
    if (!first) {
      sout << "\n\n";
    } else {
      first = false;
    }

    switch (d.getSeverity()) {
    case DiagnosticSeverity::Note:
      sout << "[NOTE]";
      break;
    case DiagnosticSeverity::Warning:
      sout << "[WARNING]";
      break;
    case DiagnosticSeverity::Error:
      sout << "[ERROR]";
      break;
    case DiagnosticSeverity::Remark:
      sout << "[REMARK]";
      break;
    default:
      sout << "[UNKNOWN]";
    }
    // Message.
    sout << ": " << d << "\n";
    printLocation(d.getLocation(), sout);
  }

  diagnostics.clear();
  return sout.str();
}

//===----------------------------------------------------------------------===//
// PyBlockRef
//===----------------------------------------------------------------------===//

void PyBlockRef::bind(py::module m) {
  py::class_<PyBlockRef>(m, "BlockRef")
      .def_property_readonly("operations",
                             [](PyBlockRef &self) {
                               return PyOperationList(
                                   self.block.getOperations());
                             })
      .def_property_readonly("args", [](PyBlockRef &self) {
        return std::vector<PyValue>(self.block.args_begin(),
                                    self.block.args_end());
      });
}

//===----------------------------------------------------------------------===//
// PyRegionRef
//===----------------------------------------------------------------------===//

void PyRegionRef::bind(py::module m) {
  py::class_<PyRegionRef>(m, "RegionRef")
      .def_property_readonly("blocks", [](PyRegionRef &self) {
        return PyBlockList(self.region.getBlocks());
      });
}

//===----------------------------------------------------------------------===//
// PyType
//===----------------------------------------------------------------------===//

void PyType::bind(py::module m) {
  py::class_<PyType>(m, "Type").def("__repr__",
                                    [](PyType &self) -> std::string {
                                      if (!self.type)
                                        return "<undefined type>";
                                      std::string res;
                                      llvm::raw_string_ostream os(res);
                                      self.type.print(os);
                                      return res;
                                    });
}

//===----------------------------------------------------------------------===//
// PyValue
//===----------------------------------------------------------------------===//

void PyValue::bind(py::module m) {
  py::class_<PyValue>(m, "Value")
      .def_property_readonly(
          "type", [](PyValue &self) -> PyType { return self.value.getType(); })
      .def("__repr__", [](PyValue &self) {
        std::string res;
        llvm::raw_string_ostream os(res);
        os << self.value;
        return res;
      });
}

//===----------------------------------------------------------------------===//
// PyAttribute
//===----------------------------------------------------------------------===//

void PyAttribute::bind(py::module m) {
  py::class_<PyAttribute>(m, "Attribute")
      .def_property_readonly("type", [](PyAttribute &self) -> PyType {
        return self.attr.getType();
      })
      .def("__repr__", [](PyAttribute &self) {
        std::string res;
        llvm::raw_string_ostream os(res);
        os << self.attr;
        return res;
      });
}

//===----------------------------------------------------------------------===//
// OpBuilder implementations
//===----------------------------------------------------------------------===//

PyBaseOpBuilder::~PyBaseOpBuilder() = default;
PyOpBuilder::~PyOpBuilder() = default;

OpBuilder &PyOpBuilder::getBuilder(bool requirePosition) {
  if (!builder.getBlock()) {
    throw py::raisePyError(PyExc_IndexError, "Insertion point not set");
  }
  return builder;
}

void PyBaseOpBuilder::bind(py::module m) {
  py::class_<PyBaseOpBuilder>(m, "BaseOpBuilder");
}

void PyOpBuilder::bind(py::module m) {
  py::class_<PyOpBuilder, PyBaseOpBuilder>(m, "OpBuilder")
      .def(py::init<PyContext &>())
      .def("clear_insertion_point",
           [](PyOpBuilder &self) { self.builder.clearInsertionPoint(); })
      .def("insert_op_before",
           [](PyOpBuilder &self, PyBaseOperation &pyOp) {
             Operation *op = pyOp.getOperation();
             self.builder.setInsertionPoint(op);
           },
           "Sets the insertion point to just before the specified op.")
      .def("insert_op_after",
           [](PyOpBuilder &self, PyBaseOperation &pyOp) {
             Operation *op = pyOp.getOperation();
             self.builder.setInsertionPointAfter(op);
           },
           "Sets the insertion point to just after the specified op.")
      .def("insert_block_start",
           [](PyOpBuilder &self, PyBlockRef block) {
             self.builder.setInsertionPointToStart(&block.block);
           },
           "Sets the insertion point to the start of the block.")
      .def("insert_block_end",
           [](PyOpBuilder &self, PyBlockRef block) {
             self.builder.setInsertionPointToEnd(&block.block);
           },
           "Sets the insertion point to the end of the block.")
      .def("insert_before_terminator",
           [](PyOpBuilder &self, PyBlockRef block) {
             auto *terminator = block.block.getTerminator();
             if (!terminator) {
               throw py::raiseValueError("Block has no terminator");
             }
             self.builder.setInsertionPoint(terminator);
           },
           "Sets the insertion point to just before the block terminator.");
}

} // namespace mlir
