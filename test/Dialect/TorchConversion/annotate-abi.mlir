// RUN: npcomp-opt -split-input-file -verify-diagnostics %s -torch-annotate-abi

// -----

// CHECK-LABEL:   builtin.func @basic_arg_and_ret(
// CHECK-SAME:    attributes {iree.abi = "{\22a\22:[\22f64\22,\22i64\22,\22i1\22],\22r\22:[\22f64\22,\22i64\22,\22i1\22],\22v\22:1}"} {
builtin.func @basic_arg_and_ret(%arg0: !torch.float, %arg1: !torch.int, %arg2: !torch.bool) -> (!torch.float, !torch.int, !torch.bool) {
  return %arg0, %arg1, %arg2 : !torch.float, !torch.int, !torch.bool
}

// -----

// CHECK-LABEL:   builtin.func @list(
// CHECK-SAME:    attributes {iree.abi = "{\22a\22:{{\[\[}}\22py_uniform_list\22,\22f64\22]],\22r\22:[],\22v\22:1}"} {
builtin.func @list(%arg0: !torch.list<!torch.float>) {
  return
}

// -----

// CHECK-LABEL:   builtin.func @tuple(
// CHECK-SAME:    attributes {iree.abi = "{\22a\22:{{\[\[}}\22pytuple\22,\22f64\22,\22i64\22]],\22r\22:[],\22v\22:1}"} {
builtin.func @tuple(%arg0: !torch.tuple<!torch.float, !torch.int>) {
  return
}

// -----

// CHECK-LABEL:   builtin.func @dict(
// CHECK-SAME:    attributes {iree.abi = "{\22a\22:{{\[\[}}\22py_uniform_dict\22,\22pystr\22,\22f64\22]],\22r\22:[],\22v\22:1}"} {
builtin.func @dict(%arg0: !torch.dict<!torch.str, !torch.float>) {
  return
}

// -----

// CHECK-LABEL:   builtin.func @tensor(
// CHECK-SAME:    attributes {iree.abi = "{\22a\22:{{\[\[}}\22ndarray\22,\22f32\22,2,null,3]],\22r\22:[],\22v\22:1}"} {
builtin.func @tensor(%arg0: !torch.tensor<[?,3],f32>) {
  return
}

// -----

// expected-error @+1 {{at function argument 0: unimplemented: ABI annotation for type '!torch.any'}}
builtin.func @unsupported(%arg0: !torch.any) {
  return
}
