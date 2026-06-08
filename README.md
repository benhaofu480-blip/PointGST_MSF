# PointGST MSF (Server Snapshot)

Synced from lab server `PointGST-main_pure` — MSF Sigmoid / Stage-2 FeedPoinTrS / PCN & KITTI tooling.

## Included

- `completion/cfgs/` — MSF, PCSA, Stage-2 feedpointrs, KITTI configs
- `completion/models/`, `Paper_related/`, `utils/`, `tools/` — core model & training
- `completion/Vision/` — visualization scripts (incl. KITTI scene vis)
- `completion/scripts/` — train/test shell helpers
- `main.py`, `requirements.txt`, top-level `cfgs/`, `tools/`, `utils/`

## Excluded (too large / not code)

- `experiments/`, `ckpt/`, `data/`, `logs/`
- `Vision/output/` PNG galleries
- `*.pth`, `*.npy` checkpoints & point caches

## Key weights (on server, not in repo)

| Stage | Path on server |
|-------|----------------|
| MSF Sigmoid PCN_core | `experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/ckpt-best.pth` |
| MSF Sigmoid PCN complete | `experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_complete/PCN_models/exp_MSF_Pure_Group_sigmoid_complete_seed42/ckpt-best.pth` |
| Stage-2 feedpointrs crop02_04 | `experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft/PCN_models/exp_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop02_04_seed42/ckpt-epoch-050.pth` |

## KITTI scene visualization

```bash
cd completion
python Vision/vis_kitti_scene_msf_complete.py --frames frame_0 --out ./kitti_vis_out
```

Uses MSF complete ckpt on server; set `--ckpt` if local path differs.
