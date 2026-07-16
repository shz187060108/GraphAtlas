param([bool]$KeepAwake = $true)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDirectory = Join-Path $ProjectRoot "outputs\logs\overnight\$timestamp"
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$latestRun = Join-Path $ProjectRoot "outputs\logs\overnight\latest_run.txt"
$latestPid = Join-Path $ProjectRoot "outputs\logs\overnight\latest_pid.txt"
$LogDirectory | Set-Content -LiteralPath $latestRun -Encoding UTF8
$worker = Join-Path $PSScriptRoot "run_overnight_experiments.ps1"
$arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $worker, "-LogDirectory", $LogDirectory, "-KeepAwake", $KeepAwake.ToString().ToLowerInvariant())
$bootstrapOut = Join-Path $LogDirectory "worker_bootstrap.stdout.log"
$bootstrapErr = Join-Path $LogDirectory "worker_bootstrap.stderr.log"
$quotedArguments = @($arguments | ForEach-Object { '"' + ([string]$_).Replace('"', '\"') + '"' })
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $quotedArguments -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden -RedirectStandardOutput $bootstrapOut -RedirectStandardError $bootstrapErr -PassThru
$process.Id | Set-Content -LiteralPath $latestPid -Encoding ASCII
$status = Join-Path $LogDirectory "status.json"
for ($attempt = 0; $attempt -lt 20 -and -not (Test-Path $status); $attempt++) { Start-Sleep -Milliseconds 500 }
if (-not (Test-Path $status)) { throw "Worker did not create status.json: $status" }
if (-not (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) {
    $state = (Get-Content $status -Raw | ConvertFrom-Json).state
    throw "Overnight worker exited during startup with state=$state."
}
Write-Host "Overnight worker PID: $($process.Id)"
Write-Host "Log directory: $LogDirectory"
Write-Host "Monitor: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\watch_overnight_experiments.ps1"
