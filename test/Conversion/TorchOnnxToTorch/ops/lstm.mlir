// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s



// CHECK-LABEL:   func.func @test_lstm_basic(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[15,2,4],f32>,
// CHECK-SAME:                               %[[VAL_1:.*]]: !torch.vtensor<[1,12,4],f32>,
// CHECK-SAME:                               %[[VAL_2:.*]]: !torch.vtensor<[1,12,3],f32>,
// CHECK-SAME:                               %[[VAL_3:.*]]: !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.ir_version = 9
// CHECK-DAG:           %[[VAL_4:.*]] = torch.constant.none
// CHECK-DAG:           %[[VAL_5:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_7:.*]] = torch.aten.select.int %[[VAL_1]], %[[VAL_5]], %[[VAL_6]]
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_10:.*]] = torch.aten.select.int %[[VAL_2]], %[[VAL_8]], %[[VAL_9]]
// CHECK-DAG:           %[[VAL_11:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_12:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_13:.*]] = torch.aten.select.int %[[VAL_3]], %[[VAL_11]], %[[VAL_12]]
// CHECK-DAG:           %[[VAL_14:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_15:.*]] = torch.constant.int 2
// CHECK-DAG:           %[[VAL_16:.*]] = torch.constant.int 3
// CHECK-DAG:           %[[VAL_17:.*]] = torch.constant.none
// CHECK-DAG:           %[[VAL_18:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_19:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_20:.*]] = torch.prim.ListConstruct %[[VAL_14]], %[[VAL_15]], %[[VAL_16]]
// CHECK-DAG:           %[[VAL_21:.*]] = torch.constant.int 6
// CHECK-DAG:           %[[VAL_22:.*]] = torch.aten.zeros %[[VAL_20]], %[[VAL_21]], %[[VAL_17]], %[[VAL_17]], %[[VAL_17]]
// CHECK-DAG:           %[[VAL_23:.*]] = torch.aten.zeros %[[VAL_20]], %[[VAL_21]], %[[VAL_17]], %[[VAL_17]], %[[VAL_17]]
// CHECK-DAG:           %[[VAL_24:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_25:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_26:.*]] = torch.aten.select.int %[[VAL_22]], %[[VAL_24]], %[[VAL_25]]
// CHECK-DAG:           %[[VAL_27:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_28:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_29:.*]] = torch.aten.select.int %[[VAL_23]], %[[VAL_27]], %[[VAL_28]]
// CHECK-DAG:           %[[VAL_30:.*]] = torch.constant.int 12
// CHECK-DAG:           %[[VAL_31:.*]] = torch.constant.int 24
// CHECK-DAG:           %[[VAL_32:.*]] = torch.aten.slice.Tensor %[[VAL_13]], %[[VAL_18]], %[[VAL_18]], %[[VAL_30]], %[[VAL_19]]
// CHECK-DAG:           %[[VAL_33:.*]] = torch.aten.slice.Tensor %[[VAL_13]], %[[VAL_18]], %[[VAL_30]], %[[VAL_31]], %[[VAL_19]]
// CHECK-DAG:           %[[VAL_34:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_35:.*]] = torch.constant.int 1
// CHECK-DAG:           %[[VAL_36:.*]] = torch.constant.int 3
// CHECK-DAG:           %[[VAL_37:.*]] = torch.constant.int 6
// CHECK-DAG:           %[[VAL_38:.*]] = torch.constant.int 9
// CHECK-DAG:           %[[VAL_39:.*]] = torch.constant.int 12
// CHECK-DAG:           %[[VAL_40:.*]] = torch.aten.slice.Tensor %[[VAL_7]], %[[VAL_34]], %[[VAL_34]], %[[VAL_36]], %[[VAL_35]]
// CHECK-DAG:           %[[VAL_41:.*]] = torch.aten.slice.Tensor %[[VAL_7]], %[[VAL_34]], %[[VAL_36]], %[[VAL_37]], %[[VAL_35]]
// CHECK-DAG:           %[[VAL_42:.*]] = torch.aten.slice.Tensor %[[VAL_7]], %[[VAL_34]], %[[VAL_37]], %[[VAL_38]], %[[VAL_35]]
// CHECK-DAG:           %[[VAL_43:.*]] = torch.aten.slice.Tensor %[[VAL_7]], %[[VAL_34]], %[[VAL_38]], %[[VAL_39]], %[[VAL_35]]
// CHECK-DAG:           %[[VAL_44:.*]] = torch.aten.slice.Tensor %[[VAL_10]], %[[VAL_34]], %[[VAL_34]], %[[VAL_36]], %[[VAL_35]]
// CHECK-DAG:           %[[VAL_45:.*]] = torch.aten.slice.Tensor %[[VAL_10]], %[[VAL_34]], %[[VAL_36]], %[[VAL_37]], %[[VAL_35]]
// CHECK-DAG:           %[[VAL_46:.*]] = torch.aten.slice.Tensor %[[VAL_10]], %[[VAL_34]], %[[VAL_37]], %[[VAL_38]], %[[VAL_35]]
// CHECK-DAG:           %[[VAL_47:.*]] = torch.aten.slice.Tensor %[[VAL_10]], %[[VAL_34]], %[[VAL_38]], %[[VAL_39]], %[[VAL_35]]
// CHECK-DAG:           %[[VAL_48:.*]] = torch.aten.slice.Tensor %[[VAL_32]], %[[VAL_34]], %[[VAL_34]], %[[VAL_36]], %[[VAL_35]]
// CHECK-DAG:           %[[VAL_49:.*]] = torch.aten.slice.Tensor %[[VAL_32]], %[[VAL_34]], %[[VAL_36]], %[[VAL_37]], %[[VAL_35]]
// CHECK-DAG:           %[[VAL_50:.*]] = torch.aten.slice.Tensor %[[VAL_32]], %[[VAL_34]], %[[VAL_37]], %[[VAL_38]], %[[VAL_35]]
// CHECK-DAG:           %[[VAL_51:.*]] = torch.aten.slice.Tensor %[[VAL_32]], %[[VAL_34]], %[[VAL_38]], %[[VAL_39]], %[[VAL_35]]
// CHECK-DAG:           %[[VAL_52:.*]] = torch.aten.slice.Tensor %[[VAL_33]], %[[VAL_34]], %[[VAL_34]], %[[VAL_36]], %[[VAL_35]]
// CHECK-DAG:           %[[VAL_53:.*]] = torch.aten.slice.Tensor %[[VAL_33]], %[[VAL_34]], %[[VAL_36]], %[[VAL_37]], %[[VAL_35]]
// CHECK-DAG:           %[[VAL_54:.*]] = torch.aten.slice.Tensor %[[VAL_33]], %[[VAL_34]], %[[VAL_37]], %[[VAL_38]], %[[VAL_35]]
// CHECK-DAG:           %[[VAL_55:.*]] = torch.aten.slice.Tensor %[[VAL_33]], %[[VAL_34]], %[[VAL_38]], %[[VAL_39]], %[[VAL_35]]
// CHECK-DAG:           %[[VAL_56:.*]] = torch.prim.ListConstruct 
// CHECK-DAG:           %[[VAL_57:.*]] = torch.constant.int 15
// CHECK-DAG:           %[[VAL_58:.*]] = torch.constant.bool true
// CHECK-DAG:           %[[VAL_59:.*]] = torch.constant.int 0
// CHECK-DAG:           %[[VAL_60:.*]]:2 = torch.prim.Loop %[[VAL_57]], %[[VAL_58]], init(%[[VAL_26]], %[[VAL_29]]) {
// CHECK-DAG:           ^bb0(%[[VAL_61:.*]]: !torch.int, %[[VAL_62:.*]]: !torch.vtensor<[2,3],f32>, %[[VAL_63:.*]]: !torch.vtensor<[2,3],f32>):
// CHECK-DAG:             %[[VAL_64:.*]] = torch.aten.select.int %[[VAL_0]], %[[VAL_59]], %[[VAL_61]]
// CHECK-DAG:             %[[VAL_65:.*]] = torch.constant.int 1
// CHECK-DAG:             %[[VAL_66:.*]] = torch.aten.linear %[[VAL_64]], %[[VAL_40]], %[[VAL_48]]
// CHECK-DAG:             %[[VAL_67:.*]] = torch.aten.linear %[[VAL_62]], %[[VAL_44]], %[[VAL_52]]
// CHECK-DAG:             %[[VAL_68:.*]] = torch.aten.add.Tensor %[[VAL_66]], %[[VAL_67]], %[[VAL_65]]
// CHECK-DAG:             %[[VAL_69:.*]] = torch.aten.sigmoid %[[VAL_68]]
// CHECK-DAG:             %[[VAL_70:.*]] = torch.aten.linear %[[VAL_64]], %[[VAL_41]], %[[VAL_49]]
// CHECK-DAG:             %[[VAL_71:.*]] = torch.aten.linear %[[VAL_62]], %[[VAL_45]], %[[VAL_53]]
// CHECK-DAG:             %[[VAL_72:.*]] = torch.aten.add.Tensor %[[VAL_70]], %[[VAL_71]], %[[VAL_65]]
// CHECK-DAG:             %[[VAL_73:.*]] = torch.aten.sigmoid %[[VAL_72]]
// CHECK-DAG:             %[[VAL_74:.*]] = torch.aten.linear %[[VAL_64]], %[[VAL_42]], %[[VAL_50]]
// CHECK-DAG:             %[[VAL_75:.*]] = torch.aten.linear %[[VAL_62]], %[[VAL_46]], %[[VAL_54]]
// CHECK-DAG:             %[[VAL_76:.*]] = torch.aten.add.Tensor %[[VAL_74]], %[[VAL_75]], %[[VAL_65]]
// CHECK-DAG:             %[[VAL_77:.*]] = torch.aten.sigmoid %[[VAL_76]]
// CHECK-DAG:             %[[VAL_78:.*]] = torch.aten.linear %[[VAL_64]], %[[VAL_43]], %[[VAL_51]]
// CHECK-DAG:             %[[VAL_79:.*]] = torch.aten.linear %[[VAL_62]], %[[VAL_47]], %[[VAL_55]]
// CHECK-DAG:             %[[VAL_80:.*]] = torch.aten.add.Tensor %[[VAL_78]], %[[VAL_79]], %[[VAL_65]]
// CHECK-DAG:             %[[VAL_81:.*]] = torch.aten.tanh %[[VAL_80]]
// CHECK-DAG:             %[[VAL_82:.*]] = torch.aten.mul.Tensor %[[VAL_77]], %[[VAL_63]]
// CHECK-DAG:             %[[VAL_83:.*]] = torch.aten.mul.Tensor %[[VAL_69]], %[[VAL_81]]
// CHECK-DAG:             %[[VAL_84:.*]] = torch.aten.add.Tensor %[[VAL_82]], %[[VAL_83]], %[[VAL_65]]
// CHECK-DAG:             %[[VAL_85:.*]] = torch.aten.tanh %[[VAL_84]]
// CHECK-DAG:             %[[VAL_86:.*]] = torch.aten.mul.Tensor %[[VAL_73]], %[[VAL_85]]
// CHECK-DAG:             %[[VAL_87:.*]] = torch.aten.append.t %[[VAL_56]], %[[VAL_86]]
// CHECK-DAG:             torch.prim.Loop.condition %[[VAL_58]], iter(%[[VAL_86]], %[[VAL_84]]
// CHECK-DAG:           }
// CHECK-DAG:           %[[VAL_88:.*]] = torch.aten.unsqueeze %[[VAL_89:.*]]#0, %[[VAL_18]]
// CHECK-DAG:           %[[VAL_90:.*]] = torch.aten.unsqueeze %[[VAL_89]]#1, %[[VAL_18]]
// CHECK-DAG:           %[[VAL_91:.*]] = torch.aten.stack %[[VAL_56]], %[[VAL_18]]
// CHECK-DAG:           %[[VAL_92:.*]] = torch.aten.unsqueeze %[[VAL_91]], %[[VAL_19]]
// CHECK-DAG:           return %[[VAL_92]], %[[VAL_88]], %[[VAL_90]]
// CHECK:         }
module {
  func.func @test_lstm_basic(%arg0: !torch.vtensor<[15,2,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[15,2,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>) -> (!torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
    return %0#0, %0#1, %0#2 : !torch.vtensor<[15,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
  }
}
