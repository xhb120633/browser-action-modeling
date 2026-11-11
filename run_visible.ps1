param(
    [ValidateSet('train', 'infer', 'compare_checkpoints')][string]$Task = 'train',
    [string]$Config = 'config.yaml'
)

# Launched in its own PowerShell window: exiting this script closes that window.
Set-Location -LiteralPath $PSScriptRoot
$Host.UI.RawUI.WindowTitle = "Jurny - $Task"
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$python = (Get-Command python -ErrorAction Stop).Source
$logDir = Join-Path $PSScriptRoot 'runs\console'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logFile = Join-Path $logDir ("$Task-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
Write-Host "Starting $Task; log: $logFile"
& $python -u "$Task.py" --config $Config 2>&1 |
    ForEach-Object { "$_" } | Tee-Object -FilePath $logFile
$result = $LASTEXITCODE
"Finished with exit code $result. Log saved to $logFile" |
    Tee-Object -FilePath $logFile -Append
@{exit_code=$result; task=$Task; log=$logFile} | ConvertTo-Json |
    Set-Content -LiteralPath ($logFile + '.status.json')
exit $result
