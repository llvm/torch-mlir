# ivalue_import

Most of the tests in this directory test importing of TorchScript
`torch::jit::Module`'s.

Modules are just one of many types of c10::IValue's and recursively contain
c10::IValue's. Thus, the work of importing TorchScript modules is mainly
about importing the wide variety of possible c10::IValue's, hence the name
of this directory and the corresponding code in ivalue_importer.cpp that it
exercises.
