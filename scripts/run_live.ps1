param(
    [string]$PythonPath,
    [double]$Threshold = 0.65,
    [switch]$AlphabetMode,
    [switch]$ShowOrigin = $true
)
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
    & $sysPy -m pip install --user -r "unified/requirements-live.txt"
    if ($LASTEXITCODE -ne 0) { throw "pip install (user site) for live deps failed." }
    $venvPy = $sysPy
  }
}

# Check for required model files
$modelPath = "unified\models\expressora_unified.tflite"
$labelsPath = "unified\models\expressora_labels.json"

if (-not (Test-Path $modelPath)) {
  Write-Host ""
  Write-Host "ERROR: Model file not found: $modelPath" -ForegroundColor Red
  Write-Host ""
  Write-Host "Please train the model first by running:" -ForegroundColor Yellow
  Write-Host "  .\scripts\run_unified.ps1" -ForegroundColor Cyan
  Write-Host ""
  Pop-Location
  exit 1
}

if (-not (Test-Path $labelsPath)) {
  Write-Host ""
  Write-Host "WARNING: Labels file not found: $labelsPath" -ForegroundColor Yellow
  Write-Host "The live script will synthesize generic labels (CLASS_0, CLASS_1, etc.)" -ForegroundColor Yellow
  Write-Host ""
}

# Install live-specific dependencies
Write-Host "Installing live inference dependencies..."
& $venvPy -m pip install -q -r "unified/requirements-live.txt"
if ($LASTEXITCODE -ne 0) {
  Write-Host "Warning: Some live dependencies may have failed to install." -ForegroundColor Yellow
  Write-Host "Continuing anyway - the script will fail if critical deps are missing." -ForegroundColor Yellow
}

# Set environment variables for thresholds and modes
$env:CONF_THRESHOLD = $Threshold.ToString()
if ($AlphabetMode) {
    $env:ALPHABET_MODE = "true"
}
if ($ShowOrigin) {
    $env:SHOW_ORIGIN = "true"
}

# Run the live camera script
Write-Host ""
Write-Host "Starting live camera inference..."
Write-Host "  Confidence Threshold: $Threshold"
if ($AlphabetMode) {
    Write-Host "  Alphabet Mode: ENABLED"
}
if ($ShowOrigin) {
    Write-Host "  Origin Display: ENABLED"
}
Write-Host ""
Write-Host "Press ESC in the camera window to quit."
Write-Host ""

& $venvPy "unified\live\live_cam_unified.py"
$exitCode = $LASTEXITCODE

Pop-Location
exit $exitCode

