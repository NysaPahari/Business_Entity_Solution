$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$neuralPython = Join-Path $PSScriptRoot '.neural-venv\Scripts\python.exe'
& $neuralPython -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $neuralPython -m pip install -r requirements.txt 'transformers>=4.48,<5'
if ($LASTEXITCODE -ne 0) { throw 'Neural dependencies failed.' }
& $neuralPython -c "import torch; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed.' }
& $neuralPython -m pip freeze | Set-Content -Encoding utf8 requirements-neural-lock.txt
