//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_JITRUNTIME_JITMODULE_H
#define NPCOMP_JITRUNTIME_JITMODULE_H

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "npcomp/RefBackend/Runtime/UserAPI.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace mlir {
class PassManager;
} // namespace mlir

namespace refback {
// Wrapper around refbackrt data structures and a JITted module, facilitating
// interaction.
class JITModule {
public:
  /// Populates a PassManager with a pipeline that performs backend compilation.
  /// The resulting module can be passed to fromCompiledModule().
  static void buildBackendCompilationPipeline(mlir::PassManager &pm,
                                              bool optimize = false);

  /// Constructs a JITModule from a compiled Module.
  /// The module should be the result of having run the backend compilation
  /// pipeline successfully.
  static llvm::Expected<std::unique_ptr<JITModule>>
  fromCompiledModule(mlir::ModuleOp module,
                     llvm::ArrayRef<llvm::StringRef> sharedLibs);

  llvm::Expected<llvm::SmallVector<refbackrt::RtValue, 6>>
  invoke(llvm::StringRef functionName,
         llvm::ArrayRef<refbackrt::RtValue> inputs);

private:
  JITModule();
  std::unique_ptr<mlir::ExecutionEngine> engine;
  refbackrt::ModuleDescriptor *descriptor;
};
} // namespace refback

#endif // NPCOMP_JITRUNTIME_JITMODULE_H
