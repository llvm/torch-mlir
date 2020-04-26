#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/CommandLine.h"
#include "npcomp/Dummy.h"

using namespace llvm;

int main(int argc, char** argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Dummy program\n");
  llvm::outs() << "Hello world!\n";
  return 0;
}
