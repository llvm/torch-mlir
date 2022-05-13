// RUN: torch-mlir-opt -torch-reduce-op-variants  -verify-diagnostics -split-input-file %s

// -----

func.func @convert_to_value_semantic_tensors_list( %list: !torch.list<tensor>) -> !torch.tensor {
  %int1 = torch.constant.int 1
  // expected-error@+1 {{failed to legalize operation 'torch.aten.cat' that was explicitly marked illegal}}
  %ret = torch.aten.cat %list, %int1 : !torch.list<tensor>, !torch.int -> !torch.tensor
  return %ret : !torch.tensor
}

// -----

func.func @convert_to_value_semantic_tensors_optional(%tensor_optional: !torch.optional<tensor>,
                                                 %t: !torch.tensor,
                                                 %training: !torch.bool,
                                                 %cudnn_enable: !torch.bool,
                                                 %f : !torch.float) -> !torch.tensor {
    // expected-error@+1 {{failed to legalize operation 'torch.aten.batch_norm' that was explicitly marked illegal}}
    %ret = torch.aten.batch_norm %t, %tensor_optional, %tensor_optional, %tensor_optional,
              %tensor_optional, %training, %f, %f, %cudnn_enable:
              !torch.tensor, !torch.optional<tensor>, !torch.optional<tensor>,
              !torch.optional<tensor>, !torch.optional<tensor>,
              !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.tensor
    return %ret: !torch.tensor
}
