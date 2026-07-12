param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$(Split-Path -Parent $MyInvocation.MyCommand.Path)\src;$env:PYTHONPATH"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

python -u scripts/run_pipeline.py --preset configs/presets/roman_p0.yaml @RemainingArgs
exit $LASTEXITCODE
