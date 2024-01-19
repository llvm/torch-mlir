# How to Add Ops to Torch-Mlir

Collected links and contacts for how to add ops to torch-mlir. 


<details>
<summary>Turbine Camp: Start Here</summary>
This document was previously known as `turbine-camp.md` to Nod.ai. "Turbine Camp" is part of Nod.ai's onboarding process. Welcome to turbine camp. This document originated at Nod.ai as a part of onboardding process, where new nod-ai folks learn about the architecture of our work by adding support for 2 ops to torch-mlir. I decided to put this into torch mlir because a lot of this is about torch-mlir. 

Written & maintained by @renxida

Guides by other folks that were used during the creation of this document:
- [Chi Liu](https://gist.github.com/AmosLewis/dd31ab37517977b1c499d06495b4adc2)
- [Sunsoon](https://docs.google.com/document/d/1H79DwW_wnVzUU81EogwY5ueXgnl-QzKet1p2lnqPar4/edit?pli=1) 

## Before you begin...

Nod-ai maintains the pipeline below, which allows us to take a ML model from e.g. huggingface, and compile it to a variety of devices including llvm-cpu, rocm and cuda and more as an optimized `vmfb` binary.

1. The pipeline begins with a huggingface model, or some other supported source like llama.cpp.
2. [nod-ai/SHARK-Turbine](https://github.com/nod-ai/SHARK-Turbine) takes a huggingface model and exports a `.mlir` file.
3. **[llvm/torch-mlir](https://github.com/llvm/torch-mlir)**, which you will be working on in turbine-camp, will lower torchscript, torch dialect, and torch aten ops further into a mixture `linalg` or `math` MLIR dialects (with occasionally other dialects in the mix)
4. [IREE](https://github.com/openxla/iree) converts the final `.mlir` file into a binary (typically `.vmfb`) for running on a device (llvm-cpu, rocm, vulcan, cuda, etc).

The details of how we do it and helpful commands to help you set up each repo is in [Sungsoon's Shark Getting Started Google Doc](https://docs.google.com/document/d/1H79DwW_wnVzUU81EogwY5ueXgnl-QzKet1p2lnqPar4/edit?pli=1)

PS: IREE is pronounced Eerie, and hence the ghost icon.

## How to begin
1. You will start by adding support for 2 ops in torch-mlir, to get you familiar with the center of our pipeline. Begin by reading [torch-mlir's documentation on how to implement a new torch op](https://github.com/llvm/torch-mlir/blob/main/docs/Torch-ops-E2E-implementation.md), and set up `llvm/torch_mlir` using https://github.com/llvm/torch-mlir/blob/main/docs/development.md
2. Pick 1 of the yet-unimplemented from the following. You should choose something that looks easy to you. **Make sure you create an issue by clicking the little "target" icon to the right of the op, thereby marking the op as yours**
    - [TorchToLinalg ops tracking issue](https://github.com/nod-ai/SHARK-Turbine/issues/347)
    - [TorchOnnnxToTorch ops tracking issue](https://github.com/nod-ai/SHARK-Turbine/issues/215)
3. Implement it. For torch -> linalg, see the how to torchop section below. For Onnx ops, see how to onnx below.
5. Make a pull request and reference your issue. When the pull request is closed, also close your issue to mark the op as done

</details>

### How to TorchToLinalg

You will need to do 4 things:
- make sure the op exists in `torch_ods_gen.py`, and then run `build_tools/update_torch_ods.sh`, and then build. This generates `GeneratedTorchOps.td`, which is used to generate the cpp and h files where ops function signatures are defined.
    - Reference [torch op registry](https://github.com/pytorch/pytorch/blob/7451dd058564b5398af79bfc1e2669d75f9ecfa2/torch/csrc/jit/passes/utils/op_registry.cpp#L21)
- make sure the op exists in `abstract_interp_lib_gen.py`, and then run `build_tools/update_abstract_interp_lib.sh`, and then build. This generates `AbstractInterpLib.cpp`, which is used to generate the cpp and h files where ops function signatures are defined.
    - Reference [torch shape functions](https://github.com/pytorch/pytorch/blob/7451dd058564b5398af79bfc1e2669d75f9ecfa2/torch/jit/_shape_functions.py#L1311)
- write test cases. They live in `projects/pt1`. See the [Dec 2023 example](https://github.com/llvm/torch-mlir/pull/2640/files).
- implement the op in one of the `lib/Conversion/TorchToLinalg/*.cpp` files

Reference Examples
- [A Dec 2023 example with the most up to date lowering](https://github.com/llvm/torch-mlir/pull/2640/files)
- [Chi's simple example of adding op lowering](https://github.com/llvm/torch-mlir/pull/1454) useful instructions and referring links for you to understand the op lowering pipeline in torch-mlir in the comments

Resources:
- how to set up torch-mlir: [https://github.com/llvm/torch-mlir/blob/main/docs/development.md](https://github.com/llvm/torch-mlir/blob/main/docs/development.md#checkout-and-build-from-source)
- torch-mlir doc on how to debug and test: [ttps://github.com/llvm/torch-mlir/blob/main/docs/development.md#testing](https://github.com/llvm/torch-mlir/blob/main/docs/development.md#testing)
- [torch op registry](https://github.com/pytorch/pytorch/blob/7451dd058564b5398af79bfc1e2669d75f9ecfa2/torch/csrc/jit/passes/utils/op_registry.cpp#L21)
- [torch shape functions](https://github.com/pytorch/pytorch/blob/7451dd058564b5398af79bfc1e2669d75f9ecfa2/torch/jit/_shape_functions.py#L1311)

### How to TorchOnnxToTorch
0. Generate the big folder of ONNX IR. Use https://github.com/llvm/torch-mlir/blob/main/test/python/onnx_importer/import_smoke_test.py . Alternatively, if you're trying to support a certain model, convert that model to onnx IR with
   ```
   optimum-cli export onnx --model facebook/opt-125M fb-opt
   python -m torch_mlir.tools.import_onnx fb-opt/model.onnx -o fb-opt-125m.onnx.mlir
   ```
2. Find an instance of the Op that you're trying to implement inside the smoke tests folder or the generated model IR, and write a test case. Later you will save it to one of the files in `torch-mlir/test/Conversion/TorchOnnxToTorch`, but for now feel free to put it anywhere.
3. Implement the op in `lib/Conversion/TorchOnnxToTorch/something.cpp`.
4. Test the conversion by running `./build/bin/torch-mlir-opt -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch your_mlir_file.mlir`. For more details, see https://github.com/llvm/torch-mlir/blob/main/docs/development.md#testing . Xida usually creates a separate MLIR file to test it to his satisfaction before integrating it into one of the files at `torch-mlir/test/Conversion/TorchOnnxToTorch`.

Helpful examples:
- [A Dec 2023 example where an ONNX op is implemented](https://github.com/llvm/torch-mlir/pull/2641/files#diff-b584b152020af6d2e5dbf62a08b2f25ed5afc2c299228383b9651d22d44b5af4R493)
- [Vivek's example of ONNX op lowering](https://github.com/llvm/torch-mlir/commit/dc9ea08db5ac295b4b3f91fc776fef6a702900b9)

## Contacts
People who've worked on this for a while
- Vivek (@vivek97 on discord)
- Chi.Liu@amd.com

Recent Turbine Camp Attendees, from recent to less recent
- Xida.ren@amd.com (@xida_ren on discord)
- Sungsoon.Cho@amd.com

## Links

- Tutorials
  - [Sungsoon's Shark Getting Started Google Doc](https://docs.google.com/document/d/1H79DwW_wnVzUU81EogwY5ueXgnl-QzKet1p2lnqPar4/edit?pli=1)
    - This document contains commands that would help you set up shark and run demos
  - [How to implement ONNX op lowering](https://github.com/llvm/torch-mlir/blob/main/docs/importers/onnx_importer.md)
- Examples
  - [A Dec 2023 example with the most up to date lowering](https://github.com/llvm/torch-mlir/pull/2640/files)
  - Chi's Example Lowering
    - Github issue and code detailing how to implement the lowring of an OP.
    - [Chi's simple example of adding op lowering](https://github.com/llvm/torch-mlir/pull/1454) useful instructions and referring links for you to understand the op lowering pipeline in torch-mlir in the comments
    - If you have questions, reach out to [Chi on Discord](https://discordapp.com/channels/973663919757492264/1104195883307892837/1180233875058868224)
  - [Vivek's example of ONNX op lowering](https://github.com/llvm/torch-mlir/commit/dc9ea08db5ac295b4b3f91fc776fef6a702900b9)
- Find Ops To Lower
  - [Torch MLIR + ONNX Unimplemented Ops on Sharepoint](https://amdcloud-my.sharepoint.com/:x:/r/personal/esaimana_amd_com/Documents/Torch%20MLIR%20+%20ONNX%20Unimplemented%20Ops.xlsx?d=w438f26fac8fd44eeafb89bc99e2c563b&csf=1&web=1&e=Qd4eHm)
    - If you don't have access yet, request it.
  - nod-ai/SHARK-Turbine ssues tracking op support
    - [Model and Op Support](https://github.com/nod-ai/SHARK-Turbine/issues/119)
    - [ONNX op support](https://github.com/nod-ai/SHARK-Turbine/issues/215)


## Chi's useful commands for debugging torch mlir

https://gist.github.com/AmosLewis/dd31ab37517977b1c499d06495b4adc2

## How to write test cases and test your new op

https://github.com/llvm/torch-mlir/blob/main/docs/development.md#testing



## How to set up vs code and intellisence for [torch-mlir]
Xida: This is optional. If you're using VS code like me, you might want to set it up so you can use the jump to definition / references, auto fix, and other features.

Feel free to contact me on discord if you have trouble figuring this out.

You may need to write something like this into your

```.vscode/settings.json```

under `torch-mlir`

```json
{
    "files.associations": {
        "*.inc": "cpp",
        "ranges": "cpp",
        "regex": "cpp",
        "functional": "cpp",
        "chrono": "cpp",
        "__functional_03": "cpp",
        "target": "cpp"
    },
    "cmake.sourceDirectory": ["/home/xida/torch-mlir/externals/llvm-project/llvm"],
    "cmake.buildDirectory": "${workspaceFolder}/build",
    "cmake.generator": "Ninja",
    "cmake.configureArgs": [
        "-DLLVM_ENABLE_PROJECTS=mlir",
        "-DLLVM_EXTERNAL_PROJECTS=\"torch-mlir\"",
        "-DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR=\"/home/xida/torch-mlir\"",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DLLVM_ENABLE_PROJECTS=mlir",
        "-DLLVM_EXTERNAL_PROJECTS=torch-mlir",
        "-DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR=${workspaceFolder}",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DLLVM_TARGETS_TO_BUILD=host",
    ],
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
    "cmake.configureEnvironment": {
        "PATH": "/home/xida/miniconda/envs/torch-mlir/bin:/home/xida/miniconda/condabin:/home/xida/miniconda/bin:/home/xida/miniconda/bin:/home/xida/miniconda/condabin:/home/xida/miniconda/bin:/home/xida/miniconda/bin:/home/xida/miniconda/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
    },
    "cmake.cmakePath": "/home/xida/miniconda/envs/torch-mlir/bin/cmake", // make sure this is a cmake that knows where your python is
}
```
The important things to note are the `cmake.configureArgs`, which specify the location of your torch mlir, and the `cmake.sourceDirectory`, which indicates that CMAKE should not build from the current directory and should instead build from `externals/llvm-project/llvm`
