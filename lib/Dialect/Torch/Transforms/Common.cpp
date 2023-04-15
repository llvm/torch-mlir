#include "Common.h"

bool getConvMiddleOps(OpList &oplist, Operation *f, int layer) {
    int convLayer = layer;
    f->walk( [&](Operation *op) { 
    if (isa<AtenConvolutionOp>(op)) {
        convLayer--;
        if(convLayer == -1) 
        oplist.insert(op);
    }
    if (convLayer == 0)
        oplist.insert(op);
    });
    // input test
    input_assert_ret(convLayer > -1, false, 
        "layer < max_layer(%d) \n", (layer-convLayer))
    return true;
}
bool getConvOp(OpList &oplist, Operation *f, int layer) {
    int convLayer = layer;
    f->walk( [&](Operation *op) { 
        if (isa<AtenConvolutionOp>(op)) {
        convLayer--;
        if (convLayer == 0)
            oplist.insert(op);
        }
    });
    // input test
    input_assert_ret(convLayer > 0, false, 
            "layer <= max_layer(%d) \n", (layer-convLayer))
    return true;
}

void creatOneTensor(vector<float> &ktensor, int64_t len) {
  for (int i = 0; i < len; i++) {
    ktensor[i * len + i] = 1.0;
  }
}
void copyTensor(std::vector<float> &ktensor, ValueTensorLiteralOp tensor) {
  for(auto i : tensor.getValue().getValues<float>()) { 
    ktensor.push_back(i); 
  }
}
