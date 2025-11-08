param([string]$PythonPath)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Cmd([object]$cmd, [string[]]$args) {
  if ($cmd -is [System.Array]) { & $cmd[0] $cmd[1] @args }
  else { & $cmd @args }
}

function Resolve-Python {
  # Prefer the Windows launcher listing
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
  # Fallback common install locations
  foreach ($p in @(
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
  )) { if (Test-Path $p) { return $p } }
  throw "No suitable Python 3.11/3.10 found. Install from https://www.python.org/downloads/windows/ (check 'Add to PATH')."
}

$py = if ($PythonPath) { $PythonPath } else { Resolve-Python }
Write-Host "Using Python: $py"

Write-Host "Creating venv (.venv)..."
& $py -m venv ".\.venv"
if ($LASTEXITCODE -ne 0) {
  Write-Host "First venv attempt failed; trying ensurepip then retry..."
  & $py -m ensurepip --upgrade
  & $py -m venv ".\.venv"
}

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
  Write-Error "Venv creation failed after retries."
  exit 1
}

$venvPy = ".\.venv\Scripts\python.exe"
Write-Host "Upgrading pip/setuptools/wheel in venv..."
& $venvPy -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
  Write-Error "pip bootstrap failed."
  exit 1
}

Write-Host "Installing unified/requirements.txt ..."
& $venvPy -m pip install -r "unified/requirements.txt"
if ($LASTEXITCODE -ne 0) {
  Write-Error "requirements install failed."
  exit 1
}

Write-Host "Venv ready at .venv"

