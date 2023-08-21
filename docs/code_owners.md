This file is a list of the people responsible for ensuring that patches for a
particular part of Torch-MLIR are reviewed, either by themself or by someone
else. They are also the gatekeepers for their part of Torch-MLIR, with the final
word on what goes in or not.

This file follows the conventions of LLVM's
[CODE_OWNERS.TXT](https://github.com/llvm/llvm-project/blob/main/llvm/CODE_OWNERS.TXT)
and Clang's
[CodeOwners.rst](https://github.com/llvm/llvm-project/blob/main/clang/CodeOwners.rst).

--------------------------------------------------------------------------------

### All parts not covered by anyone else

- Stella Laurenzo (@stellaraccident)
- Sean Silva (@silvasean) - emeritus

--------------------------------------------------------------------------------

### `torch` dialect and other core IR pieces, Python bindings/API, JIT IR importer

- Stella Laurenzo (@stellaraccident)

### TorchToLinalg, Shape inference, Dtype refinement, MaximizeValueSemantics

- Ramiro Leal-Cavazos (@ramiro050)

### CI / Build system / Packaging

- Anush Elangovan (@powderluv)

### TorchToTOSA

- Eric Kunze (@eric-k256)
- Suraj Sudhir (@sjarus)

### TorchToStablehlo

- Tianyo Kwok (@tanyokwok)
- Xiafei Qiu (@qiuxiafei)
- Ziheng Jiang (@ZihengJiang)
- Jiawei Wu (@Vremold)

### LTC

- Antonio Kim (@antoniojkim)
- Ke Deng (@ke1337)

### Bazel build

- Sambhav Jain (@sjain-stanford)
- Ahmed Taei (@asaadaldien)

### LLVM Integrate

- Ashay Rane (@ashay)
