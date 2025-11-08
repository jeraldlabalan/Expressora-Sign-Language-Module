<# 
.SYNOPSIS
  Export quantized TFLite models (FP16 and INT8)
.DESCRIPTION
  Builds representative set and exports quantized models with size/latency comparison
#>
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Resolve repo root
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $repoRoot

try {
    # Check if we're in a venv
    $inVenv = $false
    if ($env:VIRTUAL_ENV) {
        $inVenv = $true
        Write-Host "Detected active venv: $env:VIRTUAL_ENV" -ForegroundColor Green
    } else {
        $pythonPrefix = & python -c "import sys; print(sys.prefix)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $pythonPrefix -like "*\.venv*") {
            $inVenv = $true
            Write-Host "Detected active venv via sys.prefix: $pythonPrefix" -ForegroundColor Green
        }
    }
    
    if (-not $inVenv) {
        Write-Host ""
        Write-Host "WARNING: No active virtual environment detected." -ForegroundColor Yellow
        Write-Host "Please activate .venv first" -ForegroundColor Yellow
        Write-Host ""
    }
    
    # Check for SavedModel
    $savedModelPath = "unified\models\savedmodel"
    if (-not (Test-Path $savedModelPath)) {
        Write-Host ""
        Write-Host "ERROR: SavedModel not found: $savedModelPath" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please train the model first by running:" -ForegroundColor Yellow
        Write-Host "  .\scripts\run_unified.ps1" -ForegroundColor Cyan
        Write-Host ""
        exit 1
    }
    
    Write-Host ""
    Write-Host "="*60
    Write-Host "Quantized Model Export Pipeline"
    Write-Host "="*60
    Write-Host ""
    
    # Step 1: Build representative dataset
    Write-Host "Step 1: Building representative dataset..." -ForegroundColor Cyan
    & python "unified\data\build_representative_set.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: Failed to build representative dataset" -ForegroundColor Red
        exit 1
    }
    
    # Step 2: Export FP16
    Write-Host ""
    Write-Host "Step 2: Exporting FP16 quantized model..." -ForegroundColor Cyan
    & python "unified\export\export_quant_fp16.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: FP16 export failed" -ForegroundColor Red
        exit 1
    }
    
    # Step 3: Export INT8
    Write-Host ""
    Write-Host "Step 3: Exporting INT8 quantized model..." -ForegroundColor Cyan
    & python "unified\export\export_quant_int8.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: INT8 export failed" -ForegroundColor Red
        exit 1
    }
    
    # Summary: Compare model sizes
    Write-Host ""
    Write-Host "="*60
    Write-Host "Model Size Comparison"
    Write-Host "="*60
    Write-Host ""
    
    $models = @(
        @{Name="Float32 (base)"; Path="unified\models\expressora_unified.tflite"},
        @{Name="FP16 quantized"; Path="unified\models\expressora_unified_fp16.tflite"},
        @{Name="INT8 quantized"; Path="unified\models\expressora_unified_int8.tflite"}
    )
    
    $baseSize = $null
    foreach ($model in $models) {
        if (Test-Path $model.Path) {
            $size = (Get-Item $model.Path).Length
            $sizeMB = [math]::Round($size / 1MB, 2)
            
            if ($baseSize -eq $null) {
                $baseSize = $size
                $reduction = 0
            } else {
                $reduction = [math]::Round(100 * (1 - $size / $baseSize), 1)
            }
            
            $reductionStr = if ($reduction -eq 0) { "" } else { " (-$reduction%)" }
            Write-Host ("{0,-20} {1,8:F2} MB{2}" -f $model.Name, $sizeMB, $reductionStr)
        } else {
            Write-Host ("{0,-20} NOT FOUND" -f $model.Name) -ForegroundColor Yellow
        }
    }
    
    Write-Host ""
    Write-Host "="*60
    Write-Host "Quantization Complete!"
    Write-Host "="*60
    Write-Host ""
    Write-Host "All models saved to: unified\models\" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Test with live inference: .\scripts\run_live.ps1"
    Write-Host "  2. Deploy to Android: Copy .tflite + expressora_labels.json to app/src/main/assets/"
    Write-Host ""
    
} finally {
    Pop-Location
}

