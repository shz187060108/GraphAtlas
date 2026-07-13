param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
$env:PYTHONUTF8 = "1"
& python -u scripts/run_graphatlasc_suite.py @Args
