param([switch]$ShowResults)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
$previousPythonPath = $env:PYTHONPATH
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    if ($ShowResults) {
        & $pythonExecutable -m gridwatch.operations --show-results
    } else {
        & $pythonExecutable -m gridwatch.operations
    }
    if ($LASTEXITCODE -ne 0) { throw 'Operational decision laboratory failed.' }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
