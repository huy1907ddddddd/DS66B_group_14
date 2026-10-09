# A Windows junction avoids library-loader failures with non-ASCII paths.
$environmentRoot = Join-Path $PSScriptRoot '.venv'
$runtimeParent = Join-Path ([System.IO.Path]::GetPathRoot($PSScriptRoot)) 'CodexRuntimes'
$runtimeAlias = Join-Path $runtimeParent 'GridWatch_Python312'
if (-not (Test-Path -LiteralPath $environmentRoot)) { throw 'Run setup.ps1 first.' }
if (-not (Test-Path -LiteralPath $runtimeAlias)) {
    New-Item -ItemType Directory -Force -Path $runtimeParent | Out-Null
    New-Item -ItemType Junction -Path $runtimeAlias -Target $environmentRoot | Out-Null
}
$aliasTarget = (Get-Item -LiteralPath $runtimeAlias).Target
if ([System.IO.Path]::GetFullPath([string]$aliasTarget) -ne [System.IO.Path]::GetFullPath($environmentRoot)) {
    throw 'The runtime alias belongs to another checkout. Choose a different runtimeAlias in environment.ps1.'
}
$pythonExecutable = Join-Path $runtimeAlias 'Scripts\python.exe'
$env:OPENBLAS_NUM_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
