param(
    [switch]$Docker,
    [switch]$Check,
    [switch]$NoInstall,
    [switch]$Offline,
    [ValidateRange(1, 65535)][int]$Port,
    [string]$BindAddress
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$launchArgs = @((Join-Path $PSScriptRoot 'scripts\start_app.py'))
if ($Docker) { $launchArgs += '--docker' }
if ($Check) { $launchArgs += '--check' }
if ($NoInstall) { $launchArgs += '--no-install' }
if ($Offline) { $launchArgs += '--offline' }
if ($PSBoundParameters.ContainsKey('Port')) { $launchArgs += @('--port', [string]$Port) }
if ($BindAddress) { $launchArgs += @('--host', $BindAddress) }
if ($env:RAG_PYTHON) {
    & $env:RAG_PYTHON @launchArgs
} elseif (Test-Path -LiteralPath '.venv\Scripts\python.exe') {
    & '.\.venv\Scripts\python.exe' @launchArgs
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python @launchArgs
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 @launchArgs
} else {
    Write-Error 'Python 3.10+ is required. Install Python or set RAG_PYTHON.'
    exit 1
}
exit $LASTEXITCODE
