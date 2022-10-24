Write-Host "Installing python"

Start-Process winget 'install Python.Python.3.10' -wait -NoNewWindow

Write-Host "python installation completed successfully"

Write-Host "Reload environment variables"
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
Write-Host "Reloaded environment variables"

Write-Host "Installing Build Dependencies"
python -m venv .\mlir_venv\
.\mlir_venv\Scripts\activate
pip install -U pip
python.exe -m pip install --upgrade pip
pip install -r .\requirements.txt
Write-Host "Build Deps installation completed successfully"

Write-Host "Building torch-mlir"
$env:CMAKE_GENERATOR='Ninja'
$env:TORCH_MLIR_ENABLE_LTC='0'
python -m pip wheel -v -w  wheelhouse ./  -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html  -r whl-requirements.txt

Write-Host "Build completed successfully"
