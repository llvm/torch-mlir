//===- mlir_if.cpp - MLIR IR Bindings -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pybind_utils.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
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

/// Wrapper around Module, capturing a PyContext reference.
struct PyModuleOp {
  static void bind(py::module m) {
    py::class_<PyModuleOp>(m, "ModuleOp")
        .def("to_asm", &PyModuleOp::toAsm, py::arg("debug_info") = false,
             py::arg("pretty") = false, py::arg("large_element_limit") = -1);
  }

  std::string toAsm(bool enableDebugInfo, bool prettyForm,
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
    module_op.print(sout, printFlags);
    return sout.str();
  }

  std::shared_ptr<PyContext> context;
  ModuleOp module_op;
};

/// Wrapper around MLIRContext.
/// Unlike most, this is enforced to be a shared_ptr since arbitrary other
/// types can capture it.
struct PyContext : std::enable_shared_from_this<PyContext> {
  static void bind(py::module m) {
    py::class_<PyContext, std::shared_ptr<PyContext>>(m, "MLIRContext")
        .def(py::init<>([]() {
          // Need explicit make_shared to avoid UB with enable_shared_from_this.
          return std::make_shared<PyContext>();
        }))
        .def("new_module",
             [&](PyContext &context) -> PyModuleOp {
               return PyModuleOp{context.shared_from_this()};
             })
        .def("parse_asm", &PyContext::parseAsm);
  }

  PyModuleOp parseAsm(const std::string &asm_text) {
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

  MLIRContext context;
};

void defineMlirIrModule(py::module m) {
  m.doc() = "Python bindings for constructs in the mlir/IR library";

  PyContext::bind(m);
  PyModuleOp::bind(m);
}

//===----------------------------------------------------------------------===//
// Detail definitions
//===----------------------------------------------------------------------===//

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

} // namespace mlir
