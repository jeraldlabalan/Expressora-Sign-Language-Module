<# 
.SYNOPSIS
  Run evaluation on the unified model (gloss + origin classification)
.DESCRIPTION
  Assumes venv is active; installs eval dependencies and runs evaluation scripts
#>
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Resolve repo root
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $repoRoot

try {
    # Check if we're in a venv (look for VIRTUAL_ENV env var or check sys.prefix)
    $inVenv = $false
    if ($env:VIRTUAL_ENV) {
        $inVenv = $true
        Write-Host "Detected active venv: $env:VIRTUAL_ENV" -ForegroundColor Green
    } else {
        # Try to detect via Python
        $pythonPrefix = & python -c "import sys; print(sys.prefix)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $pythonPrefix -like "*\.venv*") {
            $inVenv = $true
            Write-Host "Detected active venv via sys.prefix: $pythonPrefix" -ForegroundColor Green
        }
    }
    
    if (-not $inVenv) {
        Write-Host ""
        Write-Host "WARNING: No active virtual environment detected." -ForegroundColor Yellow
        Write-Host "Please activate .venv first or run setup_venv.ps1" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Attempting to continue with current Python interpreter..." -ForegroundColor Yellow
        Write-Host ""
    }
    
    # Check for required model files
    $modelPath = "unified\models\expressora_unified.keras"
    if (-not (Test-Path $modelPath)) {
        Write-Host ""
        Write-Host "ERROR: Model file not found: $modelPath" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please train the model first by running:" -ForegroundColor Yellow
        Write-Host "  .\scripts\run_unified.ps1" -ForegroundColor Cyan
        Write-Host ""
        exit 1
    }
    
    # Install evaluation dependencies
    Write-Host ""
    Write-Host "Installing evaluation dependencies..." -ForegroundColor Cyan
    & python -m pip install -q -r "unified/requirements-eval.txt"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Some eval dependencies may have failed to install." -ForegroundColor Yellow
        Write-Host "Continuing anyway..." -ForegroundColor Yellow
    }
    
    # Run gloss evaluation
    Write-Host ""
    Write-Host "="*60
    Write-Host "Running Gloss Classification Evaluation"
    Write-Host "="*60
    Write-Host ""
    
    & python "unified\eval\eval_unified.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: Gloss evaluation failed" -ForegroundColor Red
        exit 1
    }
    
    # Run origin evaluation if origin artifacts exist
    $originPath = "unified\data\unified_origin.npy"
    $originEvalScript = "unified\eval\eval_origin.py"
    
    if ((Test-Path $originPath) -and (Test-Path $originEvalScript)) {
        Write-Host ""
        Write-Host "="*60
        Write-Host "Running Origin Classification Evaluation"
        Write-Host "="*60
        Write-Host ""
        
        & python "unified\eval\eval_origin.py"
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "WARNING: Origin evaluation failed" -ForegroundColor Yellow
            Write-Host "Continuing anyway..." -ForegroundColor Yellow
        }
    } else {
        Write-Host ""
        Write-Host "Skipping origin evaluation (origin data or script not found)" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "="*60
    Write-Host "Evaluation Complete!"
    Write-Host "="*60
    Write-Host ""
    Write-Host "Results saved to: unified\eval\" -ForegroundColor Green
    Write-Host ""
    
} finally {
    Pop-Location
}

