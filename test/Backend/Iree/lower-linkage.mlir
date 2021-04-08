// RUN: npcomp-opt -npcomp-iree-backend-lower-linkage %s | FileCheck %s

// CHECK-LABEL:   func private @decl()
func private @decl()

// CHECK-LABEL:   func @public() attributes {iree.module.export} {
func @public() {
    return
}

// CHECK-LABEL:   func private @private() {
func private @private() {
    return
}
