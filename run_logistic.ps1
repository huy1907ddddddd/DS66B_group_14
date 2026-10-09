param([switch]$ShowResults)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
$previousPythonPath = $env:PYTHONPATH
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    if ($ShowResults) {
        & $pythonExecutable -m gridwatch.logistic_alerts --show-results
    } else {
        & $pythonExecutable -m gridwatch.logistic_alerts
    }
    if ($LASTEXITCODE -ne 0) { throw 'Logistic alert explanation failed.' }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
