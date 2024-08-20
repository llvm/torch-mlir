# Uncomment if you want to install Python. GHA provides this

#Write-Host "Installing python"

#Start-Process choco 'install python --version=3.10.8' -wait -NoNewWindow

#Write-Host "python installation completed successfully"

#Write-Host "Reload environment variables"
#$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
#Write-Host "Reloaded environment variables"

Write-Host "Installing Build Dependencies"
python -m venv .\mlir_venv\
.\mlir_venv\Scripts\Activate.PS1
pip install -r .\pytorch-requirements.txt
pip install -r .\build-requirements.txt
pip install delvewheel
Write-Host "Build Deps installation completed successfully"

Write-Host "Building torch-mlir"
$env:CMAKE_GENERATOR='Ninja'
$env:TORCH_MLIR_ENABLE_LTC='0'
python -m pip wheel -v -w  wheelhouse ./  -f https://download.pytorch.org/whl/nightly/cpu/torch/  -r whl-requirements.txt

Write-Host "Build completed successfully"

Write-Host "Fixing up wheel dependencies"
delvewheel repair --add-path .\build\cmake_build\tools\torch-mlir\python_packages\torch_mlir\torch_mlir\_mlir_libs --add-dll TorchMLIRAggregateCAPI.dll --no-dll 'c10.dll;torch_python.dll;torch_cpu.dll' -v (get-item .\wheelhouse\torch_mlir*.whl).FullName
Write-Host "All Done."
