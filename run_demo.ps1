$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
. (Join-Path $projectRoot 'environment.ps1')
if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw 'Run setup.ps1 first.' }
Set-Location -LiteralPath $projectRoot
& $pythonExecutable -m streamlit run app/dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
