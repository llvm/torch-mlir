// RUN: npcomp-opt <%s -split-input-file -verify-diagnostics

// -----

torch.nn_module {
  // expected-error @+1 {{'func' op is not allowed inside `torch.nn_module`}}
  func @f()
}

// -----

torch.nn_module {
  // expected-error @+1 {{'invalidSym' does not reference a valid function}}
  torch.method "f", @invalidSym
}
