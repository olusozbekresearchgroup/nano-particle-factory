param(
    [switch]$Development,
    [switch]$SkipUpgrade
)

$ErrorActionPreference = "Stop"
$projectDir = (Split-Path -Parent $MyInvocation.MyCommand.Path)
$venvDir = Join-Path $projectDir ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

$pythonCommand = Get-Command py -ErrorAction SilentlyContinue
if ($null -eq $pythonCommand) { $pythonCommand = Get-Command python -ErrorAction SilentlyContinue }
if ($null -eq $pythonCommand) {
    throw "Python 3.10 or newer is required. Install Python and enable PATH integration."
}

if ($pythonCommand.Name -eq "py.exe") {
    & $pythonCommand.Source -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
    if ($LASTEXITCODE -ne 0) { throw "Python 3.10 or newer is required." }
    $pythonArgs = @("-3")
} else {
    & $pythonCommand.Source -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
    if ($LASTEXITCODE -ne 0) { throw "Python 3.10 or newer is required." }
    $pythonArgs = @()
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating virtual environment: $venvDir"
    & $pythonCommand.Source @pythonArgs -m venv $venvDir
}

if (-not $SkipUpgrade) {
    Write-Host "Installing build tools..."
    & $venvPython -m pip install --upgrade pip setuptools wheel
}

Write-Host "Installing Nano Particle Factory and dependencies..."
& $venvPython -m pip install --upgrade $projectDir
if ($Development) { & $venvPython -m pip install --upgrade "$projectDir[development]" }

& $venvPython -c "import ase, matplotlib, numpy, scipy, spglib, PySide6, npf; print('NPF installation verified.')"
if ($LASTEXITCODE -ne 0) { throw "NPF dependency verification failed." }

Write-Host ""
Write-Host "Installation complete."
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Start GUI with: .\.venv\Scripts\npf-gui.exe"
