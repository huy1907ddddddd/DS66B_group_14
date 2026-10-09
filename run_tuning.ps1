param([switch]$ShowResults)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
$previousPythonPath = $env:PYTHONPATH
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    if ($ShowResults) {
        & $pythonExecutable -m gridwatch.tuning --show-results
    } else {
        & $pythonExecutable -m unittest discover -s tests -p 'test_tuning_contract.py' -v
        if ($LASTEXITCODE -ne 0) { throw 'Tuning contract checks failed.' }
        & $pythonExecutable -m gridwatch.tuning
    }
    if ($LASTEXITCODE -ne 0) { throw 'Alpha tuning failed.' }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
