$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
}
. (Join-Path $projectRoot 'environment.ps1')
& $pythonExecutable -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if (-not (Test-Path -LiteralPath 'data\raw\uci_tetouan.zip')) {
    New-Item -ItemType Directory -Force -Path 'data\raw' | Out-Null
    Invoke-WebRequest -Uri 'https://archive.ics.uci.edu/static/public/849/power%2Bconsumption%2Bof%2Btetouan%2Bcity.zip' -OutFile 'data\raw\uci_tetouan.zip'
}
Write-Output 'Setup complete. Run run_experiment.ps1, then run_demo.ps1.'
