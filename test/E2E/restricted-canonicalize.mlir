// RUN: npcomp-opt -restricted-canonicalize=included-dialects=std <%s -split-input-file \
// RUN:   | FileCheck %s --check-prefix=STDONLY --dump-input=fail
// RUN: npcomp-opt -restricted-canonicalize=included-dialects=shape <%s -split-input-file \
// RUN:   | FileCheck %s --check-prefix=SHAPEONLY --dump-input=fail
// RUN: npcomp-opt -restricted-canonicalize=included-dialects=std,shape <%s -split-input-file \
// RUN:   | FileCheck %s --check-prefix=STDANDSHAPE --dump-input=fail
// RUN: not --crash npcomp-opt -restricted-canonicalize=included-dialects=notreal2,notreal1 <%s -split-input-file 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ERROR --dump-input=fail

// ERROR: restricted-canonicalize: unknown dialects: notreal1, notreal2

// STDONLY-LABEL: func @mixed_dialects
// SHAPEONLY-LABEL: func @mixed_dialects
// STDANDSHAPE-LABEL: func @mixed_dialects
func @mixed_dialects(%arg0: i32) -> i32 {

  // Do we canonicalize away the shape.assuming?
  // STDONLY: shape.assuming
  // SHAPEOONLY-NOT: shape.assuming
  // STDANDSHAPE-NOT: shape.assuming
  %w = shape.const_witness true
  %0 = shape.assuming %w -> (i32) {
    %c0 = constant 0 : i32
    shape.assuming_yield %c0 : i32
  }

  // Do we canonicalize away the std.br?
  // STDONLY-NOT: br
  // SHAPEOONLY: br
  // STDANDSHAPE-NOT: br
  br ^bb1
^bb1:
  return %0 : i32
}
