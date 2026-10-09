$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
. (Join-Path $projectRoot 'environment.ps1')
if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw 'Run setup.ps1 first.' }
$env:PYTHONPATH = Join-Path $projectRoot 'src'
Set-Location -LiteralPath $projectRoot
& $pythonExecutable -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Temporal contract checks failed.' }
& $pythonExecutable -m gridwatch.data
if ($LASTEXITCODE -ne 0) { throw 'Data audit failed.' }
& $pythonExecutable -m gridwatch.experiment
if ($LASTEXITCODE -ne 0) { throw 'Experiment failed.' }
