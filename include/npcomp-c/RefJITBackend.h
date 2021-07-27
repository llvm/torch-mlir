//===-- npcomp-c/RefJITBackend.h - C API for the reference JIT ----*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_C_REFJITBACKEND_H
#define NPCOMP_C_REFJITBACKEND_H

#include <stdbool.h>

#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// Define opaque API structs.
#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(NpcompRefJitModule, void);
DEFINE_C_API_STRUCT(NpcompRefJitValueList, void);

#undef DEFINE_C_API_STRUCT

// Must be kept in sync with C++ side.
enum NpcompRefJitElementType {
  NPCOMP_REFJIT_NONE = 0,
  NPCOMP_REFJIT_F32 = 1,
};

/// Populates a PassManager with a pipeline that performs backend compilation.
/// The resulting module can be passed to npcompRefJitModuleCreate().
MLIR_CAPI_EXPORTED void
npcompRefJitBuildBackendCompilationPipeline(MlirPassManager passManager,
                                            bool optimize);

/// Creates a RefJit module from an MlirModule (as compiled from the above
/// pipeline). On success, returns a !null NpcompRefJitModule. On failure,
/// returns null and malloc() allocates an error message into *errorMessage.
/// The caller must free these messages.
MLIR_CAPI_EXPORTED NpcompRefJitModule
npcompRefJitModuleCreate(MlirModule module, MlirStringRef *sharedLibs,
                         intptr_t sharedLibsSize, char **errorMessage);

/// Whether the module is null.
static inline bool npcompRefJitModuleIsNull(NpcompRefJitModule m) {
  return !m.ptr;
}

/// Destroys a refjit module.
MLIR_CAPI_EXPORTED void npcompRefJitModuleDestroy(NpcompRefJitModule module);

/// Invokes a function on a RefJit module. On success, returns true and malloc()
/// and adds all outputs to the passed outputs list. On failure, returns false
/// and populates *errorMessage with a malloc() allocated error message, which
/// must be caller freed.
MLIR_CAPI_EXPORTED bool
npcompRefJitModuleInvoke(NpcompRefJitModule m, MlirStringRef functionName,
                         NpcompRefJitValueList inputOutputs,
                         char **errorMessage);

/// Creates an empty value list.
MLIR_CAPI_EXPORTED NpcompRefJitValueList npcompRefJitValueListCreate();

/// Destroys a value list.
MLIR_CAPI_EXPORTED void
npcompRefJitValueListDestroy(NpcompRefJitValueList list);

/// Returns the size of the value list.
MLIR_CAPI_EXPORTED intptr_t
npcompRefJitValueListSize(NpcompRefJitValueList list);

/// Adds values to the list.
MLIR_CAPI_EXPORTED void npcompRefJitValueAddTensorCopy(
    NpcompRefJitValueList list, NpcompRefJitElementType elementType,
    const int32_t *extents, intptr_t extentsSize, const void *data);

// Reads Tensor from a list.
MLIR_CAPI_EXPORTED bool npcompRefJitValueIsaTensor(NpcompRefJitValueList list,
                                                   intptr_t i);
MLIR_CAPI_EXPORTED void *
npcompRefJitValueGetTensor(NpcompRefJitValueList list, intptr_t i,
                           NpcompRefJitElementType *elementType, intptr_t *rank,
                           const int32_t **extents);

#ifdef __cplusplus
}
#endif

#endif // NPCOMP_C_REFJITBACKEND_H
