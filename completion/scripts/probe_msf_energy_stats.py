"""Fast probe: encoder-only spectral energy stats on picked PCN test samples."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.PGST import get_basis, sort, xyz2key  # noqa: E402
from tools import builder  # noqa: E402
from vis_compare_sigmoid_vs_pcsa import CKPT_SIGMOID, _build_model  # noqa: E402


@torch.no_grad()
def main(max_samples: int = 24):
    cfg_sig = os.path.join(ROOT, "cfgs", "PCN_models", "AdaPoinTr_MSF_Pure_Group_sigmoid.yaml")
    ckpt_sig = "/tmp/ckpt-sigmoid.pth" if os.path.exists("/tmp/ckpt-sigmoid.pth") else os.path.join(ROOT, CKPT_SIGMOID)
    picks_file = os.path.join(ROOT, "data/stage1_vis_picks.txt")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print(f"ckpt={ckpt_sig}", flush=True)

    model, cfg, args = _build_model(
        cfg_sig,
        ckpt_sig,
        os.path.join(ROOT, "tmp_probe_msf_energy"),
        adapter_mode="msf_pure_group_sigmoid",
    )
    model = model.to(device).eval()
    cfg.dataset.test.others.sample_list_file = picks_file

    adapter = model.base_model.encoder.blocks.blocks[-1].gft_adapter
    eps = float(getattr(adapter, "eps", 1e-6))
    print(f"adapter={adapter.__class__.__name__}, adapt_dim={adapter.adapt_dim}, eps={eps}", flush=True)

    raw_msq_bufs = [[], []]
    energy_bufs = [[], []]

    def _make_hook(scale_idx):
        def _capture(_module, inputs, _output):
            energy = inputs[0]
            raw_msq = (energy.pow(2) - eps).clamp(min=0.0)
            raw_msq_bufs[scale_idx].append(raw_msq.detach().float().cpu().reshape(-1))
            energy_bufs[scale_idx].append(energy.detach().float().cpu().reshape(-1))
        return _capture

    h0 = adapter.energy_mlp0.register_forward_hook(_make_hook(0))
    h1 = adapter.energy_mlp1.register_forward_hook(_make_hook(1))

    _, test_loader = builder.dataset_builder(args, cfg.dataset.test)
    base = model.base_model

    n = 0
    for _taxonomy_ids, _model_ids, data in test_loader:
        partial = data[0].to(device)
        coor, f = base.grouper(partial, base.center_num)
        pe = base.pos_embed(coor)
        x = base.input_proj(f)

        b, g, _ = coor.shape
        c = coor * 100
        key = xyz2key(c[:, :, 1], c[:, :, 0], c[:, :, 2])
        _, idx0 = torch.sort(key)
        _, idx1 = torch.sort(idx0)
        sub_center = sort(coor, idx0)
        sub_u0 = get_basis(sub_center.reshape(b * (g // 16), 16, 3)).reshape(b, g // 16, 16, 16)
        sub_u1 = get_basis(sub_center.reshape(b * (g // 32), 32, 3)).reshape(b, g // 32, 32, 32)
        base.encoder(x + pe, coor, [sub_u0, sub_u1], [idx0, idx1])

        n += 1
        print(f"  sample {n}/{len(test_loader)} done", flush=True)
        if n >= max_samples:
            break

    h0.remove()
    h1.remove()

    def _summ(name, arr):
        p = np.percentile(arr, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        print(
            f"{name}: mean={arr.mean():.6f} std={arr.std():.6f} "
            f"min={arr.min():.6f} max={arr.max():.6f}"
        )
        print(
            f"  pct: p0={p[0]:.6f} p1={p[1]:.6f} p5={p[2]:.6f} p25={p[3]:.6f} p50={p[4]:.6f} "
            f"p75={p[5]:.6f} p95={p[6]:.6f} p99={p[7]:.6f} p100={p[8]:.6f}"
        )

    print(f"\nCollected from {n} samples, last encoder MSF block", flush=True)
    for scale, s in [(0, "16"), (1, "32")]:
        raw_msq = torch.cat(raw_msq_bufs[scale]).numpy()
        energy = torch.cat(energy_bufs[scale]).numpy()
        print(f"\n===== scale {s} =====")
        print("--- raw mean-square before eps: mean_k(Z'^2) ---")
        _summ("raw_msq", raw_msq)
        print("--- energy e = sqrt(raw_msq + eps), eps=1e-6 ---")
        _summ("energy", energy)
        print(f"Fraction raw_msq < eps: {(raw_msq < eps).mean()*100:.4f}%")
        rel = np.sqrt((raw_msq + eps) / np.maximum(raw_msq, 1e-12)) - 1.0
        print("Relative inflation sqrt(raw+eps)/sqrt(raw)-1:")
        _summ("rel_inflation", rel)
        for alt_eps in [1e-8, 1e-6, 1e-4]:
            e_alt = np.sqrt(raw_msq + alt_eps)
            diff = np.abs(e_alt - energy).mean()
            print(f"mean |e(eps={alt_eps:.0e}) - e(eps=1e-6)| = {diff:.8f}")


if __name__ == "__main__":
    main(max_samples=12)
