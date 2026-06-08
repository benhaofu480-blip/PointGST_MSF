# Pre-single_MSF_trial backup (2026-06-01)

Restore original optimizer behavior:

```bash
cp tools/backup/pre_single_msf_trial/builder.py tools/builder.py
```

Files backed up:
- `tools/builder.py` — original `gft` / `decoder` / `only_new` paths unchanged
- `cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_complete.yaml` — baseline Stage-1 config

New experiment (does not modify backed-up files in place):
- `tools/freeze_policy.py` — `gft_single_decoder` policy
- `cfgs/PCN_models/AdaPoinTr_MSF_single_MSF_trial.yaml`
- `scripts/train_single_MSF_trial.sh`
