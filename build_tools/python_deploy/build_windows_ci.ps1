#Uncomment if you want to test locally. GHA provides Python
#Write-Host "Installing python"

#Start-Process choco 'install python --version=3.10.8' -wait -NoNewWindow

#Write-Host "python installation completed successfully"

#Write-Host "Reload environment variables"
#$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
#Write-Host "Reloaded environment variables"

Write-Host "Installing Build Dependencies"
python -m venv .\mlir_venv\
.\mlir_venv\Scripts\activate
pip install -r .\requirements.txt
Write-Host "Build Deps installation completed successfully"

Write-Host "Building torch-mlir"

Start-Process cmake '-GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir;torch-mlir-dialects" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR="$PWD"/externals/llvm-external-projects/torch-mlir-dialects \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  /externals/llvm-project/llvm'

Write-Host "Build completed successfully"

Write-Host "Testing torch-mlir"
$env:PYTHONPATH = "$PWD/build/tools/torch-mlir/python_packages/torch_mlir;$PWD/examples"
cmake --build build --target check-torch-mlir-all
Write-Host "Testing completed successfully"
