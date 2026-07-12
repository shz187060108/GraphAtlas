param([Parameter(ValueFromRemainingArguments=$true)][string[]]$RemainingArgs)
$ErrorActionPreference="Stop"; $env:PYTHONUTF8="1"; $root=Split-Path -Parent $MyInvocation.MyCommand.Path; Set-Location $root
python -u scripts/run_pipeline.py --preset configs/presets/atlas_v2_screening.yaml @RemainingArgs
exit $LASTEXITCODE
