param(
    [string]$LogDirectory,
    [string]$KeepAwake = "true"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "C:\Users\24953\.conda\envs\dl\python.exe"
if (-not $LogDirectory) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogDirectory = Join-Path $ProjectRoot "outputs\logs\overnight\$timestamp"
}
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$PipelineLog = Join-Path $LogDirectory "pipeline.log"
$StatusPath = Join-Path $LogDirectory "status.json"
$FailurePath = Join-Path $LogDirectory "failure_summary.txt"
$script:CompletedStages = @()
$script:Commands = @()
$script:ExitCodes = @{}
$script:CurrentStage = "initializing"
$script:StartedAt = (Get-Date).ToString("o")
$KeepAwakeEnabled = $KeepAwake.ToLowerInvariant() -notin @("false", "0", "no")

function Write-Status([string]$State, [string]$FailedStage = $null) {
    $payload = [ordered]@{
        state = $State
        current_stage = $script:CurrentStage
        completed_stages = @($script:CompletedStages)
        failed_stage = $FailedStage
        started_at = $script:StartedAt
        updated_at = (Get-Date).ToString("o")
        pid = $PID
        commands = @($script:Commands)
        exit_codes = $script:ExitCodes
        log_directory = $LogDirectory
    }
    $temporary = "$StatusPath.tmp"
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $StatusPath -Force
}

function Invoke-Logged([string]$Stage, [string[]]$Arguments, [string]$Stem) {
    $script:CurrentStage = $Stage
    $commandText = '"{0}" {1}' -f $Python, ($Arguments -join ' ')
    $script:Commands += $commandText
    Write-Status "running"
    Add-Content -LiteralPath $PipelineLog -Value "[$(Get-Date -Format o)] START $Stage`n$commandText"
    $stdout = Join-Path $LogDirectory "$Stem.stdout.log"
    $stderr = Join-Path $LogDirectory "$Stem.stderr.log"
    $quotedArguments = @($Arguments | ForEach-Object { '"' + ([string]$_).Replace('"', '\"') + '"' })
    $process = Start-Process -FilePath $Python -ArgumentList $quotedArguments -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru -WindowStyle Hidden
    $code = $process.ExitCode
    $script:ExitCodes[$Stage] = $code
    if (Test-Path $stdout) { Get-Content $stdout | Add-Content $PipelineLog }
    if (Test-Path $stderr) { Get-Content $stderr | Add-Content $PipelineLog }
    Add-Content -LiteralPath $PipelineLog -Value "[$(Get-Date -Format o)] END $Stage exit=$code"
    if ($code -ne 0) {
        $tail = if (Test-Path $stderr) { (Get-Content $stderr -Tail 80) -join "`n" } else { "No stderr log." }
        "Stage: $Stage`nExit code: $code`n`n$tail" | Set-Content -LiteralPath $FailurePath -Encoding UTF8
        Write-Status "failed" $Stage
        throw "$Stage failed with exit code $code"
    }
    $script:CompletedStages += $Stage
    Write-Status "running"
}

if ($KeepAwakeEnabled) {
    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class AwakeState {
  [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
}
"@
    [void][AwakeState]::SetThreadExecutionState([uint32]2147483649)
}

try {
    Write-Status "running"
    Invoke-Logged "python_version" @("--version") "python_version"
    Invoke-Logged "environment" @("-X", "utf8", "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA'); assert torch.cuda.is_available(), 'CUDA is required for overnight experiments'") "environment"
    Invoke-Logged "suite_help" @("-u", "scripts\run_graphatlasc_suite.py", "--help") "suite_help"

    if (-not $env:H2GB_PYTHON) { throw "H2GB_PYTHON is not set for the detached worker." }
    Invoke-Logged "h2gb_convert" @("scripts\download_h2gb.py", "--datasets", "pdns", "mag-year", "ieee-cis", "--source-root", "data\h2gb_official", "--output-root", "data\h2gb", "--h2gb-python", $env:H2GB_PYTHON, "--offline") "h2gb_convert"
    Invoke-Logged "h2gb_check" @("scripts\check_h2gb.py", "--root", "data\h2gb") "h2gb_check"
    Invoke-Logged "h2gb_dry_run" @("-u", "scripts\run_graphatlasc_suite.py", "--stage", "h2gb", "--dry-run") "h2gb_dry_run"
    Invoke-Logged "real_dry_run" @("-u", "scripts\run_graphatlasc_suite.py", "--stage", "real", "--dry-run") "real_dry_run"

    $manifestPath = Join-Path $ProjectRoot "outputs\manifests\graphatlasc_real_screening.jsonl"
    if (-not (Test-Path $manifestPath)) { throw "The current real dry-run did not create $manifestPath" }
    $datasets = Get-Content $manifestPath | ForEach-Object { ($_ | ConvertFrom-Json).dataset } | Sort-Object -Unique
    $datasets | Set-Content -LiteralPath (Join-Path $LogDirectory "manifest_datasets.txt") -Encoding UTF8
    $required = @("actor", "questions", "dblp", "coauthor_cs", "coauthor_physics", "hetgb_texas", "hetgb_actor", "hetgb_amazon")
    $excluded = @("amazon_ratings", "minesweeper")
    $missing = @($required | Where-Object { $_ -notin $datasets })
    $presentExcluded = @($excluded | Where-Object { $_ -in $datasets })
    if ($missing.Count -or $presentExcluded.Count) {
        $message = "Real manifest validation failed.`nMissing: $($missing -join ', ')`nExcluded but present: $($presentExcluded -join ', ')"
        $message | Set-Content -LiteralPath $FailurePath -Encoding UTF8
        throw $message
    }
    $script:CompletedStages += "real_manifest_validation"
    Write-Status "running"

    Invoke-Logged "real" @("-X", "utf8", "-u", "scripts\run_graphatlasc_suite.py", "--stage", "real") "real"
    Invoke-Logged "confirmatory" @("-X", "utf8", "-u", "scripts\run_graphatlasc_suite.py", "--stage", "confirmatory") "confirmatory"
    Invoke-Logged "ogb" @("-X", "utf8", "-u", "scripts\run_graphatlasc_suite.py", "--stage", "ogb") "ogb"
    Invoke-Logged "figures" @("-u", "scripts\build_paper_figures.py", "--results", "outputs\best_config_search\search_summary.csv", "--output-dir", "outputs\reports\paper_figures", "--target-model", "graphatlas_c_oracle") "figures"
    $script:CurrentStage = "done"
    Write-Status "completed"
}
catch {
    if (-not (Test-Path $FailurePath)) { $_ | Out-String | Set-Content -LiteralPath $FailurePath -Encoding UTF8 }
    if ((Get-Content $StatusPath -Raw | ConvertFrom-Json).state -ne "failed") { Write-Status "failed" $script:CurrentStage }
    exit 1
}
finally {
    if ($KeepAwakeEnabled) { [void][AwakeState]::SetThreadExecutionState([uint32]2147483648) }
}
