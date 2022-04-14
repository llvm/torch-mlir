// RUN: torch-mlir-opt <%s -convert-torch-to-mhlo -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL:   func.func @torch.aten.convolution(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !torch.vtensor<[7,3,16,16],f32>) -> !torch.vtensor<[7,6,6,16],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[7,3,16,16],f32> -> tensor<7x3x16x16xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = mhlo.constant dense<"0xF260193E0FB23E3E33D12BBC8D07AF3D42A618BE8044C3BCFAD235BE500922BED83E203ED7CAF0BD45C82ABE95A1293E3340B23B1AE8393D2733863D7AF55D3DF54817BE0030C3BB7F1C11BE676D91BDE3AD4BBE4D33963CF79A0DBE8013A8BC4054BCBDA20E123E808F3A3C87B311BD33D0F1BD8DB3FF3DB0B70DBEC0C9103DDF8A443E071B7C3D5A2E7C3DC063DCBC339D24BD3398DD3CE2F34B3E1F7E1D3EE791353ECDEB713DCA93F7BDF342F23DDA5330BD8A9228BE5208363E500B913D00DABE3D87B87F3D873D07BDD3AE1BBEFAE100BDE0F2B4BD6D1906BE0258273EF32EACBC873B883DD5B015BE339501BDF316EB3D0BA640BE87248E3D2000C0BD87C61EBE122A203EBA773DBE57A684BDF3A1D7BC4DE4F93C00B37DBC8D8F013DA70E70BD2AC3D4BD4D41023DFA9AC73DDAA101BE332093BCA31DE7BD40AB69BD6769A9BD138F94BDF3B8E43C87E4153D93DA8D3D95183DBEB7AFDBBDF347FEBDE069A53DAD1C4BBE5AC2EBBC9767913DEDBE20BDBA82BC3D0AE2A2BD0A47863DDA17973D9A6D46BE6BEC3CBE67189D3B9D5A83BD5F5201BECD4BD63D801F5F3D236EBFBD87F0BB3D873A3D3E8785A73D937F393EC0B537BD3A8A0ABE40969E3C3AA93FBEBA7B2CBEA30CADBDFB62273E3AD6E33D7AEE403D0F3F1CBE878E6ABDE3DB263EA0E5433DBA78F0BD701582BD8754BE3D3D8745BEFA0A13BDCD5293BC2745A0BC00F85F3C00358FBCD35AD1BD40AE153EA7CEECBCF3841F3E5AF480BC109F91BD90BC2FBEAAE1C4BDD399FE3DD2891EBE8385EB3D0D98A63DADFF943D83C3AEBD37B24CBE33243A3D77D9E73D9726D2BD6DA5D83D"> : tensor<6x1x5x5xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_8:.*]] = torch_c.to_i64 %[[VAL_7]]
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_14:.*]] = mhlo.convolution(%[[VAL_1]], %[[VAL_3]]) 
// CHECK-SAME{LITERAL}:                                                       dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 1], pad = [[4, 4], [2, 2]], rhs_dilate = [3, 1]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64} : (tensor<7x3x16x16xf32>, tensor<6x1x5x5xf32>) -> tensor<7x6x6x16xf32>
// CHECK:           %[[VAL_15:.*]] = torch_c.from_builtin_tensor %[[VAL_14]] : tensor<7x6x6x16xf32> -> !torch.vtensor<[7,6,6,16],f32>
// CHECK:           return %[[VAL_15]] : !torch.vtensor<[7,6,6,16],f32>
func.func @torch.aten.convolution(%arg0: !torch.vtensor<[7,3,16,16],f32>) -> !torch.vtensor<[7,6,6,16],f32> {
  %none = torch.constant.none
  %0 = torch.vtensor.literal(dense<"0xF260193E0FB23E3E33D12BBC8D07AF3D42A618BE8044C3BCFAD235BE500922BED83E203ED7CAF0BD45C82ABE95A1293E3340B23B1AE8393D2733863D7AF55D3DF54817BE0030C3BB7F1C11BE676D91BDE3AD4BBE4D33963CF79A0DBE8013A8BC4054BCBDA20E123E808F3A3C87B311BD33D0F1BD8DB3FF3DB0B70DBEC0C9103DDF8A443E071B7C3D5A2E7C3DC063DCBC339D24BD3398DD3CE2F34B3E1F7E1D3EE791353ECDEB713DCA93F7BDF342F23DDA5330BD8A9228BE5208363E500B913D00DABE3D87B87F3D873D07BDD3AE1BBEFAE100BDE0F2B4BD6D1906BE0258273EF32EACBC873B883DD5B015BE339501BDF316EB3D0BA640BE87248E3D2000C0BD87C61EBE122A203EBA773DBE57A684BDF3A1D7BC4DE4F93C00B37DBC8D8F013DA70E70BD2AC3D4BD4D41023DFA9AC73DDAA101BE332093BCA31DE7BD40AB69BD6769A9BD138F94BDF3B8E43C87E4153D93DA8D3D95183DBEB7AFDBBDF347FEBDE069A53DAD1C4BBE5AC2EBBC9767913DEDBE20BDBA82BC3D0AE2A2BD0A47863DDA17973D9A6D46BE6BEC3CBE67189D3B9D5A83BD5F5201BECD4BD63D801F5F3D236EBFBD87F0BB3D873A3D3E8785A73D937F393EC0B537BD3A8A0ABE40969E3C3AA93FBEBA7B2CBEA30CADBDFB62273E3AD6E33D7AEE403D0F3F1CBE878E6ABDE3DB263EA0E5433DBA78F0BD701582BD8754BE3D3D8745BEFA0A13BDCD5293BC2745A0BC00F85F3C00358FBCD35AD1BD40AE153EA7CEECBCF3841F3E5AF480BC109F91BD90BC2FBEAAE1C4BDD399FE3DD2891EBE8385EB3D0D98A63DADFF943D83C3AEBD37B24CBE33243A3D77D9E73D9726D2BD6DA5D83D"> : tensor<6x1x5x5xf32>) : !torch.vtensor<[6,1,5,5],f32>
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %int3 = torch.constant.int 3
  %1 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int4, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %false = torch.constant.bool false
  %5 = torch.aten.convolution %arg0, %0, %none, %1, %2, %3, %false, %4, %int3 : !torch.vtensor<[7,3,16,16],f32>, !torch.vtensor<[6,1,5,5],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[7,6,6,16],f32>
  return %5 : !torch.vtensor<[7,6,6,16],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.convolution$bias(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[7,3,16,16],f32>) -> !torch.vtensor<[7,6,6,16],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[7,3,16,16],f32> -> tensor<7x3x16x16xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<[0.0907487422, -0.018881727, 0.0906107202, 0.111919597, -0.19894366, 1.406200e-01]> : tensor<6xf32>
// CHECK:           %[[VAL_3:.*]] = mhlo.constant dense<"0xCDD2DC3CB7A7E23DBD93093EC8A01C3E9A08603D27E50A3DB735C2BD730B15BE23380FBEA0F73BBD40A1A63D80568EBC3BD0253ED82007BEB7F1413E00F4A9BD67B5B93C9AF0223E00732FBE6AD7DA3DC7EECBBD1BCA033ECB0428BEAD13F23D0078B1BBED537D3DFD832BBEBDC0BE3D1722AE3D53C2733DB5C53E3E404E13BE4D8B9EBC9A3E933CDD6FFBBD075DE63DA75CAA3C4D2EC73C87F379BD074C3ABD0052D9BC730BCCBD8D596ABD475E24BE4A29193ECD10283D809135BE27A1D83CD33BB8BD4BB2353E337B62BA4D240EBEFFD502BEE71B22BEFAE48ABD3366C3BDB720D03D3A567FBD025545BE828028BE52D425BE5D21A43D52550FBE105D06BE4A68E1BD53CD2B3EBDBDB63D9ADAFCBDE0F11E3EB5C32D3E1AC02A3DFAC733BDBBD746BE20243ABE2D941E3DE0314C3DC22F493E50791B3E12A116BED3C5C23DD56610BE2D15213D2066713D8A6A453E8DC1ECBD83ACC5BDA39D83BD58EA49BE5ABC35BE8ADA003EFA56263D1DCDF63DDAC64ABD9074AD3D13D630BE9333E43DE09A823DC0FFA13C3DF8B6BD6D3F88BDC52920BEF365C6BD137F2CBE97913A3ECDA9F3BDBD84903D979CA9BD80DB693DC03A313DB30C19BE3566273E53F82F3E8A9845BE6A23C73DA8411CBEED0A213E67D53B3C63AD22BE755411BE6BB223BEC7ACA2BDCDF4B43A9A33073E333F26BC27F37C3D57BEDE3D5AE341BE7383CEBC2D9CAC3DA3361EBE73D196BC0A182E3EB34DC5BCAD3062BDB7988CBD75CD193EE0A2B3BDBF950C3E0D9F833D338CD6BDD75134BE4D92D63D1ADDD9BD6F534B3EC2BF1BBECDDCA9B9CD4E263CBD914ABE525A42BE10E9B93D"> : tensor<6x1x5x5xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_8:.*]] = torch_c.to_i64 %[[VAL_7]]
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_14:.*]] = mhlo.convolution(%[[VAL_1]], %[[VAL_3]]) 
// CHECK-SAME{LITERAL}:                                                       dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 1], pad = [[4, 4], [2, 2]], rhs_dilate = [3, 1]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64} : (tensor<7x3x16x16xf32>, tensor<6x1x5x5xf32>) -> tensor<7x6x6x16xf32>
// CHECK:           %[[VAL_15:.*]] = mhlo.constant dense<[1, 6, 1, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_16:.*]] = "mhlo.dynamic_reshape"(%[[VAL_2]], %[[VAL_15]]) : (tensor<6xf32>, tensor<4xi64>) -> tensor<1x6x1x1xf32>
// CHECK:           %[[VAL_17:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_16]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x6x1x1xf32>) -> tensor<7x6x6x16xf32>
// CHECK:           %[[VAL_18:.*]] = mhlo.add %[[VAL_14]], %[[VAL_17]] : tensor<7x6x6x16xf32>
// CHECK:           %[[VAL_19:.*]] = torch_c.from_builtin_tensor %[[VAL_18]] : tensor<7x6x6x16xf32> -> !torch.vtensor<[7,6,6,16],f32>
// CHECK:           return %[[VAL_19]] : !torch.vtensor<[7,6,6,16],f32>
func.func @torch.aten.convolution$bias(%arg0: !torch.vtensor<[7,3,16,16],f32>) -> !torch.vtensor<[7,6,6,16],f32> {
  %0 = torch.vtensor.literal(dense<[0.0907487422, -0.018881727, 0.0906107202, 0.111919597, -0.19894366, 1.406200e-01]> : tensor<6xf32>) : !torch.vtensor<[6],f32>
  %1 = torch.vtensor.literal(dense<"0xCDD2DC3CB7A7E23DBD93093EC8A01C3E9A08603D27E50A3DB735C2BD730B15BE23380FBEA0F73BBD40A1A63D80568EBC3BD0253ED82007BEB7F1413E00F4A9BD67B5B93C9AF0223E00732FBE6AD7DA3DC7EECBBD1BCA033ECB0428BEAD13F23D0078B1BBED537D3DFD832BBEBDC0BE3D1722AE3D53C2733DB5C53E3E404E13BE4D8B9EBC9A3E933CDD6FFBBD075DE63DA75CAA3C4D2EC73C87F379BD074C3ABD0052D9BC730BCCBD8D596ABD475E24BE4A29193ECD10283D809135BE27A1D83CD33BB8BD4BB2353E337B62BA4D240EBEFFD502BEE71B22BEFAE48ABD3366C3BDB720D03D3A567FBD025545BE828028BE52D425BE5D21A43D52550FBE105D06BE4A68E1BD53CD2B3EBDBDB63D9ADAFCBDE0F11E3EB5C32D3E1AC02A3DFAC733BDBBD746BE20243ABE2D941E3DE0314C3DC22F493E50791B3E12A116BED3C5C23DD56610BE2D15213D2066713D8A6A453E8DC1ECBD83ACC5BDA39D83BD58EA49BE5ABC35BE8ADA003EFA56263D1DCDF63DDAC64ABD9074AD3D13D630BE9333E43DE09A823DC0FFA13C3DF8B6BD6D3F88BDC52920BEF365C6BD137F2CBE97913A3ECDA9F3BDBD84903D979CA9BD80DB693DC03A313DB30C19BE3566273E53F82F3E8A9845BE6A23C73DA8411CBEED0A213E67D53B3C63AD22BE755411BE6BB223BEC7ACA2BDCDF4B43A9A33073E333F26BC27F37C3D57BEDE3D5AE341BE7383CEBC2D9CAC3DA3361EBE73D196BC0A182E3EB34DC5BCAD3062BDB7988CBD75CD193EE0A2B3BDBF950C3E0D9F833D338CD6BDD75134BE4D92D63D1ADDD9BD6F534B3EC2BF1BBECDDCA9B9CD4E263CBD914ABE525A42BE10E9B93D"> : tensor<6x1x5x5xf32>) : !torch.vtensor<[6,1,5,5],f32>
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %int3 = torch.constant.int 3
  %2 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int4, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct %int3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %5 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %false = torch.constant.bool false
  %6 = torch.aten.convolution %arg0, %1, %0, %2, %3, %4, %false, %5, %int3 : !torch.vtensor<[7,3,16,16],f32>, !torch.vtensor<[6,1,5,5],f32>, !torch.vtensor<[6],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[7,6,6,16],f32>
  return %6 : !torch.vtensor<[7,6,6,16],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.convolution$dynamic(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>, 
// CHECK-SAME:                                 %[[VAL_3:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_16:.*]] = torch_c.to_builtin_tensor %[[VAL_3]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_7:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_8:.*]] = torch_c.to_i64 %[[VAL_7]]
// CHECK:           %[[VAL_9:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_10:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_11:.*]] = torch.prim.ListConstruct %[[VAL_7]], %[[VAL_5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_14:.*]] = mhlo.convolution(%[[VAL_1]], %[[VAL_16]]) 
// CHECK-SAME{LITERAL}:                                                        dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 1], pad = [[4, 4], [2, 2]], rhs_dilate = [3, 1]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_15:.*]] = torch_c.from_builtin_tensor %[[VAL_14]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[VAL_15]] : !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten.convolution$dynamic(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %none = torch.constant.none
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %int3 = torch.constant.int 3
  %1 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int4, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %false = torch.constant.bool false
  %5 = torch.aten.convolution %arg0, %arg1, %none, %1, %2, %3, %false, %4, %int3 : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,?],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
  return %5 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.mm$basic(
// CHECK-SAME:                             %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>,
// CHECK-SAME:                             %[[VAL_1:.*]]: !torch.vtensor<[64,32],f32>) -> !torch.vtensor<[4,32],f32> {
// CHECK:          %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:          %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[64,32],f32> -> tensor<64x32xf32>
// CHECK:          %[[VAL_4:.*]] = "mhlo.dot"(%[[VAL_2]], %[[VAL_3]]) : (tensor<4x64xf32>, tensor<64x32xf32>) -> tensor<4x32xf32>
// CHECK:          %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<4x32xf32> -> !torch.vtensor<[4,32],f32>
// CHECK:          return %[[VAL_5]] : !torch.vtensor<[4,32],f32>
func.func @torch.aten.mm$basic(%arg0: !torch.vtensor<[4,64],f32>, %arg1: !torch.vtensor<[64,32],f32>) -> !torch.vtensor<[4,32],f32> {
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[4,64],f32>, !torch.vtensor<[64,32],f32> -> !torch.vtensor<[4,32],f32>
  return %0 : !torch.vtensor<[4,32],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.bmm$basic(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[8,3,4],f32>,
// CHECK-SAME:                               %[[VAL_1:.*]]: !torch.vtensor<[8,4,5],f32>) -> !torch.vtensor<[8,3,5],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[8,3,4],f32> -> tensor<8x3x4xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[8,4,5],f32> -> tensor<8x4x5xf32>
// CHECK:           %[[VAL_4:.*]] = "mhlo.dot_general"(%[[VAL_2]], %[[VAL_3]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<8x3x4xf32>, tensor<8x4x5xf32>) -> tensor<8x3x5xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<8x3x5xf32> -> !torch.vtensor<[8,3,5],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[8,3,5],f32>
func.func @torch.aten.bmm$basic(%arg0: !torch.vtensor<[8,3,4],f32>, %arg1: !torch.vtensor<[8,4,5],f32>) -> !torch.vtensor<[8,3,5],f32> {
  %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[8,3,4],f32>, !torch.vtensor<[8,4,5],f32> -> !torch.vtensor<[8,3,5],f32>
  return %0 : !torch.vtensor<[8,3,5],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.matmul$mat_vec(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[2,3],f32>, %[[VAL_1:.*]]: !torch.vtensor<[3],f32>) -> !torch.vtensor<[2],f32> {
// CHECK:         %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,3],f32> -> tensor<2x3xf32>
// CHECK:         %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[3],f32> -> tensor<3xf32>
// CHECK:         %[[VAL_4:.*]] = "mhlo.dot"(%[[VAL_2]], %[[VAL_3]]) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2xf32>
// CHECK:         %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<2xf32> -> !torch.vtensor<[2],f32>
// CHECK:         return %[[VAL_5]] : !torch.vtensor<[2],f32>
func.func @torch.aten.matmul$mat_vec(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[3],f32>) -> !torch.vtensor<[2],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3],f32> -> !torch.vtensor<[2],f32>
  return %0 : !torch.vtensor<[2],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.matmul$vec_mat(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[3],f32>, %[[VAL_1:.*]]: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[5],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3],f32> -> tensor<3xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[3,5],f32> -> tensor<3x5xf32>
// CHECK:           %[[VAL_4:.*]] = "mhlo.dot_general"(%[[VAL_2]], %[[VAL_3]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<3xf32>, tensor<3x5xf32>) -> tensor<5xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<5xf32> -> !torch.vtensor<[5],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[5],f32>
func.func @torch.aten.matmul$vec_mat(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[5],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3,5],f32> -> !torch.vtensor<[5],f32>
  return %0 : !torch.vtensor<[5],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.matmul$batch_mat_vec(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[2,3,5],f32>, %[[VAL_1:.*]]: !torch.vtensor<[5],f32>) -> !torch.vtensor<[2,3],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,3,5],f32> -> tensor<2x3x5xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[5],f32> -> tensor<5xf32>
// CHECK:           %[[VAL_4:.*]] = mhlo.constant dense<[1, 5, 1]> : tensor<3xi64>
// CHECK:           %[[VAL_5:.*]] = "mhlo.dynamic_reshape"(%[[VAL_3]], %[[VAL_4]]) : (tensor<5xf32>, tensor<3xi64>) -> tensor<1x5x1xf32>
// CHECK:           %[[VAL_6:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_5]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x5x1xf32>) -> tensor<2x5x1xf32>
// CHECK:           %[[VAL_7:.*]] = "mhlo.dot_general"(%[[VAL_2]], %[[VAL_6]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<2x3x5xf32>, tensor<2x5x1xf32>) -> tensor<2x3x1xf32>
// CHECK:           %[[VAL_8:.*]] = mhlo.constant dense<[2, 3]> : tensor<2xi64>
// CHECK:           %[[VAL_9:.*]] = "mhlo.dynamic_reshape"(%[[VAL_7]], %[[VAL_8]]) : (tensor<2x3x1xf32>, tensor<2xi64>) -> tensor<2x3xf32>
// CHECK:           %[[VAL_10:.*]] = torch_c.from_builtin_tensor %[[VAL_9]] : tensor<2x3xf32> -> !torch.vtensor<[2,3],f32>
// CHECK:           return %[[VAL_10]] : !torch.vtensor<[2,3],f32>
func.func @torch.aten.matmul$batch_mat_vec(%arg0: !torch.vtensor<[2,3,5],f32>, %arg1: !torch.vtensor<[5],f32>) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[2,3,5],f32>, !torch.vtensor<[5],f32> -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.matmul$batch_mat_mat(
// CEHCK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[2,1,3,5],f32>, %[[VAL_1:.*]]: !torch.vtensor<[4,5,7],f32>) -> !torch.vtensor<[2,4,3,7],f32> {
// CEHCK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,1,3,5],f32> -> tensor<2x1x3x5xf32>
// CEHCK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,5,7],f32> -> tensor<4x5x7xf32>
// CEHCK:           %[[VAL_4:.*]] = mhlo.constant dense<[1, 4, 5, 7]> : tensor<4xi64>
// CEHCK:           %[[VAL_5:.*]] = "mhlo.dynamic_reshape"(%[[VAL_3]], %[[VAL_4]]) : (tensor<4x5x7xf32>, tensor<4xi64>) -> tensor<1x4x5x7xf32>
// CEHCK:           %[[VAL_6:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_2]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<2x1x3x5xf32>) -> tensor<2x4x3x5xf32>
// CEHCK:           %[[VAL_7:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_5]]) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x4x5x7xf32>) -> tensor<2x4x5x7xf32>
// CEHCK:           %[[VAL_8:.*]] = "mhlo.dot_general"(%[[VAL_6]], %[[VAL_7]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]>} : (tensor<2x4x3x5xf32>, tensor<2x4x5x7xf32>) -> tensor<2x4x3x7xf32>
// CEHCK:           %[[VAL_9:.*]] = mhlo.constant dense<[2, 4, 3, 7]> : tensor<4xi64>
// CEHCK:           %[[VAL_10:.*]] = "mhlo.dynamic_reshape"(%[[VAL_8]], %[[VAL_9]]) : (tensor<2x4x3x7xf32>, tensor<4xi64>) -> tensor<2x4x3x7xf32>
// CEHCK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_10]] : tensor<2x4x3x7xf32> -> !torch.vtensor<[2,4,3,7],f32>
// CEHCK:           return %[[VAL_11]] : !torch.vtensor<[2,4,3,7],f32>
func.func @torch.aten.matmul$batch_mat_mat(%arg0: !torch.vtensor<[2,1,3,5],f32>, %arg1: !torch.vtensor<[4,5,7],f32>) -> !torch.vtensor<[2,4,3,7],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[2,1,3,5],f32>, !torch.vtensor<[4,5,7],f32> -> !torch.vtensor<[2,4,3,7],f32>
  return %0 : !torch.vtensor<[2,4,3,7],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.linear$bias(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,5],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,3],f32> -> tensor<2x3xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<[0.332985669, -0.351702571, 0.497120917, 0.284032375, -0.28389287]> : tensor<5xf32>
// CHECK:           %[[VAL_3:.*]] = mhlo.constant 
// CHECK-SAME{LITERAL}                            dense<[[0.0281541087, 0.0428751968, 0.41462335], [-0.00366509636, -0.168656304, 0.420785218], [0.307634622, -0.285770357, -0.455667704], [0.524494112, 0.522319436, 0.542355359], [-0.0946185663, -0.223030448, -0.523420036]]> : tensor<5x3xf32>
// CHECK:           %[[VAL_4:.*]] = "mhlo.dot_general"(%[[VAL_1]], %[[VAL_3]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>} : (tensor<2x3xf32>, tensor<5x3xf32>) -> tensor<2x5xf32>
// CHECK:           %[[VAL_5:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_2]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<5xf32>) -> tensor<2x5xf32>
// CHECK:           %[[VAL_6:.*]] = mhlo.add %[[VAL_4]], %[[VAL_5]] : tensor<2x5xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<2x5xf32> -> !torch.vtensor<[2,5],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[2,5],f32>
func.func @torch.aten.linear$bias(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,5],f32> {
  %0 = torch.vtensor.literal(dense<[0.332985669, -0.351702571, 0.497120917, 0.284032375, -0.28389287]> : tensor<5xf32>) : !torch.vtensor<[5],f32>
  %1 = torch.vtensor.literal(dense<[[0.0281541087, 0.0428751968, 0.41462335], [-0.00366509636, -0.168656304, 0.420785218], [0.307634622, -0.285770357, -0.455667704], [0.524494112, 0.522319436, 0.542355359], [-0.0946185663, -0.223030448, -0.523420036]]> : tensor<5x3xf32>) : !torch.vtensor<[5,3],f32>
  %2 = torch.aten.linear %arg0, %1, %0 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[5,3],f32>, !torch.vtensor<[5],f32> -> !torch.vtensor<[2,5],f32>
  return %2 : !torch.vtensor<[2,5],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.linear(
// CHECK-SAME:                            %[[VAL_0:.*]]: !torch.vtensor<[2,4,3],f32>) -> !torch.vtensor<[2,4,5],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,4,3],f32> -> tensor<2x4x3xf32>
// CHECK:           %none = torch.constant.none
// CHECK:           %[[VAL_2:.*]] = mhlo.constant 
// CHECK-SAME{LITERAL}                            dense<[[-0.398664087, -0.0268191695, -0.36585173], [-0.538538814, 0.390200675, 0.0402345695], [0.113372549, -0.3134287, 0.236918554], [-0.12469662, -0.214788809, 0.219961867], [0.0569517352, 0.464073777, -0.542487681]]> : tensor<5x3xf32>
// CHECK:           %[[VAL_3:.*]] = "mhlo.dot_general"(%[[VAL_1]], %[[VAL_2]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<2x4x3xf32>, tensor<5x3xf32>) -> tensor<2x4x5xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<2x4x5xf32> -> !torch.vtensor<[2,4,5],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[2,4,5],f32>
func.func @torch.aten.linear(%arg0: !torch.vtensor<[2,4,3],f32>) -> !torch.vtensor<[2,4,5],f32> {
  %none = torch.constant.none
  %0 = torch.vtensor.literal(dense<[[-0.398664087, -0.0268191695, -0.36585173], [-0.538538814, 0.390200675, 0.0402345695], [0.113372549, -0.3134287, 0.236918554], [-0.12469662, -0.214788809, 0.219961867], [0.0569517352, 0.464073777, -0.542487681]]> : tensor<5x3xf32>) : !torch.vtensor<[5,3],f32>
  %1 = torch.aten.linear %arg0, %0, %none : !torch.vtensor<[2,4,3],f32>, !torch.vtensor<[5,3],f32>, !torch.none -> !torch.vtensor<[2,4,5],f32>
  return %1 : !torch.vtensor<[2,4,5],f32>
}