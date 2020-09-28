// RUN: npcomp-opt %s | FileCheck %s

func @dummy() {
  // CHECK: "torch.dummy"
  "torch.dummy"() : () -> ()
  return
}
