%6 = torch.aten._make_per_tensor_quantized_tensor %5, %float1.572600e-02, %int63 : !torch.vtensor<[1,16],ui8>, !torch.float, !torch.int -> !torch.vtensor<[1,16],!torch.quint8>
%10 = torch.aten._make_per_tensor_quantized_tensor %3, %9, %int0 : !torch.vtensor<[8,16],si8>, !torch.float, !torch.int -> !torch.vtensor<[8,16],!torch.qint8>
%11 = torch.aten.dequantize.self %10 : !torch.vtensor<[8,16],!torch.qint8> -> !torch.vtensor<[8,16],f32>
%16 = torch.aten.transpose.int %11, %int0, %int1 : !torch.vtensor<[8,16],f32>, !torch.int, !torch.int -> !torch.vtensor<[16,8],f32>
%17 = torch.aten.mm %6, %16 : !torch.vtensor<[1,16],!torch.quint8>, !torch.vtensor<[16,8],f32> -> !torch.vtensor<[1,8],f32>

to


%6 = torch.aten._make_per_tensor_quantized_tensor %5, %float1.572600e-02, %int63 : !torch.vtensor<[1,16],ui8>, !torch.float, !torch.int -> !torch.vtensor<[1,16],!torch.quint8>
%16 = torch.aten.transpose.int %3, %int0, %int1 : !torch.vtensor<[8,16],si8>, !torch.int, !torch.int -> !torch.vtensor<[16,8],si8>
%10 = torch.aten._make_per_tensor_quantized_tensor %16, %9, %int0 : !torch.vtensor<[16,8],si8>, !torch.float, !torch.int -> !torch.vtensor<[16,8],!torch.qint8>
%17 = torch.aten.mm %6, %10 : !torch.vtensor<[1,16],!torch.quint8>, !torch.vtensor<[16,8],!torch.qint8> -> !torch.vtensor<[1,8],f32>

// mm operand -> is transpose?
// transpose operand -> is dequant?
// get dequant operand (should be _make_per_tensor_quantized_tensor)
// get the operand from that (here %3)
// Value newTranspose = rewriter.create<transpose>(op->getLoc(), result type = (shape from transpose op, elt type from quant operand), input = %3 (quant operand), other transpose args )
// Value newMPTQT = rewriter.create<MPTQT>(op->getLoc(), resultType = (shape from transpose op, elt type from dequant operand), input = newTranspose, other MPTQT args );
// operand[i] = newMPTQT;
// rewriter.replaceOpWithNewOp<SrcOp>(op, operands);
// unused ops will get cleaned up eventually. 
