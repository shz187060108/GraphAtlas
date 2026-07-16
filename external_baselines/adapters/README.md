# Optional external adapters

External baselines are deliberately not cloned or emulated. Set `NSD_ROOT` or
`BUNN_ROOT` to an existing compatible checkout, then use
`scripts/run_external_baselines.py`; unavailable dependencies are recorded as
unavailable unless `--require-external` is requested.
