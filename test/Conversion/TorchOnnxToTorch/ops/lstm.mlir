// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s



// CHECK-LABEL:   func.func @test_lstm_basic(
// CHECK-SAME:                    %[[VAL_0:.*]]: !torch.vtensor<[15,2,4],f32>,
// CHECK-SAME:                    %[[VAL_1:.*]]: !torch.vtensor<[1,12,4],f32>,
// CHECK-SAME:                    %[[VAL_2:.*]]: !torch.vtensor<[1,12,3],f32>,
// CHECK-SAME:                    %[[VAL_3:.*]]: !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
// CHECK:           %[[VAL_4:.*]] = torch.constant.none
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_7:.*]] = torch.aten.select.int %[[VAL_1]], %[[VAL_5]], %[[VAL_6]] : !torch.vtensor<[1,12,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,4],f32>
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_10:.*]] = torch.aten.select.int %[[VAL_2]], %[[VAL_8]], %[[VAL_9]] : !torch.vtensor<[1,12,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,3],f32>
// CHECK:           %[[VAL_11:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_12:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_13:.*]] = torch.aten.select.int %[[VAL_3]], %[[VAL_11]], %[[VAL_12]] : !torch.vtensor<[1,24],f32>, !torch.int, !torch.int -> !torch.vtensor<[24],f32>
// CHECK:           %[[VAL_14:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_15:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_16:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_17:.*]] = torch.constant.none
// CHECK:           %[[VAL_18:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_19:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_20:.*]] = torch.prim.ListConstruct %[[VAL_14]], %[[VAL_15]], %[[VAL_16]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_21:.*]] = torch.constant.int 6
// CHECK:           %[[VAL_22:.*]] = torch.aten.zeros %[[VAL_20]], %[[VAL_21]], %[[VAL_17]], %[[VAL_17]], %[[VAL_17]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[1,2,3],f32>
// CHECK:           %[[VAL_23:.*]] = torch.aten.zeros %[[VAL_20]], %[[VAL_21]], %[[VAL_17]], %[[VAL_17]], %[[VAL_17]] : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[1,2,3],f32>
// CHECK:           %[[VAL_24:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_25:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_26:.*]] = torch.aten.select.int %[[VAL_22]], %[[VAL_24]], %[[VAL_25]] : !torch.vtensor<[1,2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
// CHECK:           %[[VAL_27:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_28:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_29:.*]] = torch.aten.select.int %[[VAL_23]], %[[VAL_27]], %[[VAL_28]] : !torch.vtensor<[1,2,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
// CHECK:           %[[VAL_30:.*]] = torch.constant.int 12
// CHECK:           %[[VAL_31:.*]] = torch.constant.int 24
// CHECK:           %[[VAL_32:.*]] = torch.aten.slice.Tensor %[[VAL_13]], %[[VAL_18]], %[[VAL_18]], %[[VAL_30]], %[[VAL_19]] : !torch.vtensor<[24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[12],f32>
// CHECK:           %[[VAL_33:.*]] = torch.aten.slice.Tensor %[[VAL_13]], %[[VAL_18]], %[[VAL_30]], %[[VAL_31]], %[[VAL_19]] : !torch.vtensor<[24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[12],f32>
// CHECK:           %[[VAL_34:.*]] = torch.prim.ListConstruct  : () -> !torch.list<vtensor<[2,3],f32>>
// CHECK:           %[[VAL_35:.*]] = torch.constant.int 15
// CHECK:           %[[VAL_36:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_37:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_38:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_39:.*]]:2 = torch.prim.Loop %[[VAL_35]], %[[VAL_36]], init(%[[VAL_26]], %[[VAL_29]]) {
// CHECK:           ^bb0(%[[VAL_40:.*]]: !torch.int, %[[VAL_41:.*]]: !torch.vtensor<[2,3],f32>, %[[VAL_42:.*]]: !torch.vtensor<[2,3],f32>):
// CHECK:             %[[VAL_43:.*]] = torch.aten.select.int %[[VAL_0]], %[[VAL_37]], %[[VAL_40]] : !torch.vtensor<[15,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,4],f32>
// CHECK:             %[[VAL_44:.*]] = torch.constant.int 0
// CHECK:             %[[VAL_45:.*]] = torch.constant.int 1
// CHECK:             %[[VAL_46:.*]] = torch.constant.int 4
// CHECK:             %[[VAL_47:.*]] = torch.prim.ListConstruct %[[VAL_45]], %[[VAL_46]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAL_48:.*]] = torch.aten.tile %[[VAL_43]], %[[VAL_47]] : !torch.vtensor<[2,4],f32>, !torch.list<int> -> !torch.vtensor<[2,16],f32>
// CHECK:             %[[VAL_49:.*]] = torch.aten.tile %[[VAL_41]], %[[VAL_47]] : !torch.vtensor<[2,3],f32>, !torch.list<int> -> !torch.vtensor<[2,12],f32>
// CHECK:             %[[VAL_50:.*]] = torch.aten.linear %[[VAL_48]], %[[VAL_7]], %[[VAL_32]] : !torch.vtensor<[2,16],f32>, !torch.vtensor<[12,4],f32>, !torch.vtensor<[12],f32> -> !torch.vtensor<[2,12],f32>
// CHECK:             %[[VAL_51:.*]] = torch.aten.linear %[[VAL_49]], %[[VAL_10]], %[[VAL_33]] : !torch.vtensor<[2,12],f32>, !torch.vtensor<[12,3],f32>, !torch.vtensor<[12],f32> -> !torch.vtensor<[2,12],f32>
// CHECK:             %[[VAL_52:.*]] = torch.aten.add.Tensor %[[VAL_50]], %[[VAL_51]], %[[VAL_45]] : !torch.vtensor<[2,12],f32>, !torch.vtensor<[2,12],f32>, !torch.int -> !torch.vtensor<[2,12],f32>
// CHECK:             %[[VAL_53:.*]] = torch.constant.int 3
// CHECK:             %[[VAL_54:.*]] = torch.constant.int 6
// CHECK:             %[[VAL_55:.*]] = torch.constant.int 9
// CHECK:             %[[VAL_56:.*]] = torch.constant.int 12
// CHECK:             %[[VAL_57:.*]] = torch.aten.slice.Tensor %[[VAL_52]], %[[VAL_45]], %[[VAL_44]], %[[VAL_55]], %[[VAL_45]] : !torch.vtensor<[2,12],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,9],f32>
// CHECK:             %[[VAL_58:.*]] = torch.aten.slice.Tensor %[[VAL_52]], %[[VAL_45]], %[[VAL_55]], %[[VAL_56]], %[[VAL_45]] : !torch.vtensor<[2,12],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
// CHECK:             %[[VAL_59:.*]] = torch.aten.sigmoid %[[VAL_57]] : !torch.vtensor<[2,9],f32> -> !torch.vtensor<[2,9],f32>
// CHECK:             %[[VAL_60:.*]] = torch.aten.tanh %[[VAL_58]] : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>
// CHECK:             %[[VAL_61:.*]] = torch.aten.slice.Tensor %[[VAL_59]], %[[VAL_45]], %[[VAL_44]], %[[VAL_53]], %[[VAL_45]] : !torch.vtensor<[2,9],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
// CHECK:             %[[VAL_62:.*]] = torch.aten.slice.Tensor %[[VAL_59]], %[[VAL_45]], %[[VAL_53]], %[[VAL_54]], %[[VAL_45]] : !torch.vtensor<[2,9],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
// CHECK:             %[[VAL_63:.*]] = torch.aten.slice.Tensor %[[VAL_59]], %[[VAL_45]], %[[VAL_54]], %[[VAL_55]], %[[VAL_45]] : !torch.vtensor<[2,9],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
// CHECK:             %[[VAL_64:.*]] = torch.aten.mul.Tensor %[[VAL_63]], %[[VAL_42]] : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>
// CHECK:             %[[VAL_65:.*]] = torch.aten.mul.Tensor %[[VAL_61]], %[[VAL_60]] : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>
// CHECK:             %[[VAL_66:.*]] = torch.aten.add.Tensor %[[VAL_64]], %[[VAL_65]], %[[VAL_45]] : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[2,3],f32>
// CHECK:             %[[VAL_67:.*]] = torch.aten.tanh %[[VAL_66]] : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>
// CHECK:             %[[VAL_68:.*]] = torch.aten.mul.Tensor %[[VAL_62]], %[[VAL_67]] : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>
// CHECK:             %[[VAL_69:.*]] = torch.aten.append.t %[[VAL_34]], %[[VAL_68]] : !torch.list<vtensor<[2,3],f32>>, !torch.vtensor<[2,3],f32> -> !torch.list<vtensor<[2,3],f32>>
// CHECK:             %[[VAL_70:.*]] = torch.aten.add.int %[[VAL_40]], %[[VAL_38]] : !torch.int, !torch.int -> !torch.int
// CHECK:             torch.prim.Loop.condition %[[VAL_36]], iter(%[[VAL_68]], %[[VAL_66]] : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>)
// CHECK:           } : (!torch.int, !torch.bool, !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>) -> (!torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>)
// CHECK:           %[[VAL_71:.*]] = torch.aten.unsqueeze %[[VAL_72:.*]]#0, %[[VAL_18]] : !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[1,2,3],f32>
// CHECK:           %[[VAL_73:.*]] = torch.aten.unsqueeze %[[VAL_72]]#1, %[[VAL_18]] : !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[1,2,3],f32>
// CHECK:           %[[VAL_74:.*]] = torch.aten.stack %[[VAL_34]], %[[VAL_18]] : !torch.list<vtensor<[2,3],f32>>, !torch.int -> !torch.vtensor<[15,2,3],f32>
// CHECK:           %[[VAL_75:.*]] = torch.aten.unsqueeze %[[VAL_74]], %[[VAL_19]] : !torch.vtensor<[15,2,3],f32>, !torch.int -> !torch.vtensor<[15,1,2,3],f32>
// CHECK:           return %[[VAL_75]], %[[VAL_71]], %[[VAL_73]] : !torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
// CHECK:         }
module {
  func.func @test_lstm_basic(%arg0: !torch.vtensor<[15,2,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[15,2,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
    return %0#0, %0#1, %0#2 : !torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
  }
}