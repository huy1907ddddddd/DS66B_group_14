param([switch]$ShowResults)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
$previousPythonPath = $env:PYTHONPATH
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    if ($ShowResults) {
        & $pythonExecutable -m gridwatch.forest_tuning --show-results
    } else {
        & $pythonExecutable -m gridwatch.forest_tuning
    }
    if ($LASTEXITCODE -ne 0) { throw 'Random Forest selection failed.' }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
