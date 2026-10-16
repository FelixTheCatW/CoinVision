#requires -Version 7
param(
    [switch]$Install
)

$ErrorActionPreference = "Stop"

Write-Host "CoinVision bootstrap" -ForegroundColor Cyan

Write-Host "GPU:" -ForegroundColor Yellow
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

Write-Host "`nPython 3.12 required (PyTorch CUDA 12.6 wheels, Pascal sm_61)." -ForegroundColor Yellow

if ($Install) {
    uv sync --extra dev
    Write-Host "`nVerifying CUDA availability..." -ForegroundColor Yellow
    uv run python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('arches', torch.cuda.get_arch_list() if torch.cuda.is_available() else [])"
}
