param([string]$PythonPath)
<# requires Windows PowerShell 5.1+ #>
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Cmd([object]$cmd, [string[]]$args) {
  if ($cmd -is [System.Array]) { & $cmd[0] $cmd[1] @args }
  else { & $cmd @args }
}

function Resolve-SystemPython {
  $paths = & py -0p 2>$null
  if ($LASTEXITCODE -eq 0 -and $paths) {
    $lines = $paths -split "`r?`n" | Where-Object { $_ -match "python\.exe" }
    foreach ($want in @("3.11","3.10")) {
      $match = $lines | Where-Object { $_ -match "\-V:$want" } | Select-Object -First 1
      if ($match) {
        if ($match -match "([A-Z]:\\.*python\.exe)") { return $Matches[1] }
      }
    }
  }
  foreach ($p in @(
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
  )) { if (Test-Path $p) { return $p } }
  throw "No working system Python 3.11/3.10 found."
}

function Test-PythonWorks([string]$pyPathOrCmd) {
  try {
    $ver = & $pyPathOrCmd -c "import sys;print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>&1
    if ($LASTEXITCODE -ne 0) { return $false }
    $v = [version]$ver
    if (!($v.Major -eq 3 -and $v.Minor -le 11)) { return $false }
    $out = & $pyPathOrCmd -c "print('ok')" 2>&1
    if ($LASTEXITCODE -ne 0) { return $false }
    return ($out -match "ok") -and -not ($out -match "Unable to create process using")
  } catch { return $false }
}

# Resolve repo root based on this script's folder
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $repoRoot

$venvPy = ".\.venv\Scripts\python.exe"
$useVenv = (Test-Path $venvPy) -and (Test-PythonWorks $venvPy)

if (-not $useVenv) {
  Write-Host "Creating venv via setup_venv.ps1 ..."
  if ($PythonPath) {
    & (Join-Path $PSScriptRoot "setup_venv.ps1") -PythonPath $PythonPath
  } else {
    & (Join-Path $PSScriptRoot "setup_venv.ps1")
  }
  if ($LASTEXITCODE -eq 0 -and (Test-Path $venvPy) -and (Test-PythonWorks $venvPy)) {
    $useVenv = $true
  } else {
    Write-Host "Venv creation failed; falling back to system Python (<=3.11) and user-site install."
    $sysPy = if ($PythonPath) { $PythonPath } else { Resolve-SystemPython }
    & $sysPy -m pip install --user -r "unified/requirements.txt"
    if ($LASTEXITCODE -ne 0) { throw "pip install (user site) failed." }
    $venvPy = $sysPy
  }
}

Write-Host "Running unified pipeline..."
$pyCmd = $venvPy
& $pyCmd tools\check_python.py
if ($LASTEXITCODE -ne 0) { throw "check_python failed" }
& $pyCmd unified\data\build_unified_dataset.py
if ($LASTEXITCODE -ne 0) { throw "build_unified_dataset failed" }
& $pyCmd unified\training\train_unified_tf.py
if ($LASTEXITCODE -ne 0) { throw "train_unified_tf failed" }
& $pyCmd unified\export\export_unified_tflite.py
if ($LASTEXITCODE -ne 0) { throw "export_unified_tflite failed" }
& $pyCmd unified\inference\quick_test_tflite.py
if ($LASTEXITCODE -ne 0) { throw "quick_test_tflite failed" }

# Generate concept-key mapping (non-fatal)
Write-Host ""
Write-Host "Generating concept-key mapping..." -ForegroundColor Cyan
& $pyCmd unified\bridge\apply_label_map.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Label mapping failed (non-critical)" -ForegroundColor Yellow
}

# Generate model card (non-fatal)
Write-Host ""
Write-Host "Generating model card..." -ForegroundColor Cyan
& $pyCmd unified\export\write_model_card.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Model card generation failed (non-critical)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host 'Done. Artifacts should be under unified\models' -ForegroundColor Green
Pop-Location
