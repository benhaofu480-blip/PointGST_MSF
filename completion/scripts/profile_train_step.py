#!/usr/bin/env python3
"""Profile one PCN training step with torch.profiler.

Usage (from completion/):
  # full train step (forward + loss + backward + optim)
  CUDA_VISIBLE_DEVICES=0 python scripts/profile_train_step.py \\
    --config cfgs/PCN_models/AdaPoinTr_MSF_single_MSF_trial.yaml \\
    --ckpt ckpt/AdaPoinTr_ps55.pth \\
    --tag single_MSF_trial

  # forward-only: isolate inference cost (decoder still runs even if frozen)
  CUDA_VISIBLE_DEVICES=0 python scripts/profile_train_step.py \\
    --config cfgs/PCN_models/AdaPoinTr_MSF_single_MSF_trial.yaml \\
    --ckpt ckpt/AdaPoinTr_ps55.pth \\
    --tag single_MSF_trial_fwd \\
    --forward-only
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from tools import builder  # noqa: E402
from utils.config import cfg_from_yaml_file  # noqa: E402


class _RecordFunctionHook:
    def __init__(self, name: str):
        self.name = name
        self._ctx = None

    def pre(self, *_args, **_kwargs):
        self._ctx = record_function(self.name)
        self._ctx.__enter__()

    def post(self, *_args, **_kwargs):
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None


def _install_module_hooks(model: nn.Module, fine_grained: bool = False):
    hooks = []
    targets = [
        ("base_model.grouper", model.base_model.grouper),
        ("base_model.encoder", model.base_model.encoder),
        ("base_model.coarse_stack", None),
        ("base_model.decoder", model.base_model.decoder),
        ("decode_head", model.decode_head),
        ("increase_dim", model.increase_dim),
        ("reduce_map", model.reduce_map),
    ]
    coarse_modules = [
        model.base_model.pos_embed,
        model.base_model.input_proj,
        model.base_model.increase_dim,
        model.base_model.coarse_pred,
        model.base_model.mem_link,
        model.base_model.query_ranking,
        model.base_model.mlp_query,
    ]

    def _hook_pair(name, module):
        rf = _RecordFunctionHook(name)
        hooks.append(module.register_forward_pre_hook(rf.pre))
        hooks.append(module.register_forward_hook(rf.post))

    for name, module in targets:
        if module is None:
            continue
        _hook_pair(name, module)

    if fine_grained:
        enc_blocks = model.base_model.encoder.blocks.blocks
        for i, block in enumerate(enc_blocks):
            _hook_pair(f"encoder.block{i}", block)
        dec_blocks = model.base_model.decoder.blocks.blocks
        for i, block in enumerate(dec_blocks):
            _hook_pair(f"decoder.block{i}", block)

        class _CoarseStackHook:
            def pre(self, *_args, **_kwargs):
                self._ctx = record_function("base_model.coarse_stack")
                self._ctx.__enter__()

            def post(self, *_args, **_kwargs):
                if self._ctx is not None:
                    self._ctx.__exit__(None, None, None)
                    self._ctx = None

        coarse_rf = _CoarseStackHook()
        for mod in coarse_modules:
            hooks.append(mod.register_forward_pre_hook(coarse_rf.pre))
            hooks.append(mod.register_forward_hook(coarse_rf.post))

    return hooks


def _aggregate_keywords(prof, keywords):
    buckets = defaultdict(float)
    for evt in prof.key_averages():
        name = evt.key
        cuda_us = evt.cuda_time_total
        if cuda_us <= 0:
            continue
        matched = False
        for bucket, kws in keywords.items():
            if any(k.lower() in name.lower() for k in kws):
                buckets[bucket] += cuda_us
                matched = True
                break
        if not matched:
            buckets["other"] += cuda_us
    total = sum(buckets.values()) or 1.0
    return {k: (v, 100.0 * v / total) for k, v in sorted(buckets.items(), key=lambda x: -x[1])}


def _one_step(model, partial, gt, optimizer, forward_only=False):
    if not forward_only:
        optimizer.zero_grad(set_to_none=True)
    with record_function("forward_total"):
        ret = model(partial)
    with record_function("loss_total"):
        sparse_loss, dense_loss = model.module.get_loss(ret, gt)
        loss = sparse_loss + dense_loss
    if forward_only:
        return float(loss.item())
    with record_function("backward_total"):
        loss.backward()
    with record_function("optimizer_step"):
        optimizer.step()
    return float(loss.item())


def _collect_record_function_rows(prof, prefix_filter=None):
    rows = []
    for evt in prof.key_averages():
        key = evt.key
        if prefix_filter and not key.startswith(prefix_filter):
            continue
        if evt.cuda_time_total <= 0:
            continue
        rows.append((evt.cuda_time_total, key, evt.count))
    return sorted(rows, reverse=True)


def _write_module_breakdown(buf, prof, forward_only):
    buf.write("\n--- Module / phase labels (record_function) ---\n")
    module_rows = _collect_record_function_rows(prof)
    module_rows = [
        (us, key, cnt)
        for us, key, cnt in module_rows
        if key.startswith("base_model.")
        or key.endswith("_total")
        or key.startswith("encoder.block")
        or key.startswith("decoder.block")
        or key in {"decode_head", "increase_dim", "reduce_map"}
    ]
    for us, key, cnt in module_rows:
        buf.write(f"  {key:32s}  cuda_total={us/1000:.2f} ms  count={cnt}\n")

    dec_rows = [(us, key, cnt) for us, key, cnt in module_rows if key.startswith("decoder.block")]
    if dec_rows:
        dec_total = sum(us for us, _, _ in dec_rows)
        buf.write(f"\n  decoder blocks sum: {dec_total/1000:.2f} ms\n")
        for us, key, cnt in sorted(dec_rows, key=lambda x: x[0], reverse=True):
            pct = 100.0 * us / dec_total if dec_total else 0.0
            buf.write(f"    {key:32s}  {us/1000:6.2f} ms  ({pct:4.1f}% of decoder)\n")

    phase_keys = ["forward_total", "loss_total", "backward_total", "optimizer_step"]
    phase_us = {}
    for evt in prof.key_averages():
        if evt.key in phase_keys:
            phase_us[evt.key] = evt.cuda_time_total
    if phase_us:
        buf.write("\n--- Phase totals (cuda) ---\n")
        for key in phase_keys:
            if key in phase_us:
                buf.write(f"  {key:18s}  {phase_us[key]/1000:8.2f} ms\n")
        if forward_only and "forward_total" in phase_us and "loss_total" in phase_us:
            fwd = phase_us["forward_total"] + phase_us["loss_total"]
            buf.write(f"  {'forward+loss':18s}  {fwd/1000:8.2f} ms\n")

    if forward_only:
        buf.write("\n--- Decoder fully frozen: what changes? ---\n")
        buf.write(
            "  Forward: decoder must still run (query refinement). "
            "Freezing weights does NOT skip these matmuls.\n"
        )
        buf.write(
            "  Backward: MSF/rebuild-head still need grad w.r.t. mem through frozen decoder, "
            "so backward through decoder layers remains (~same GEMM cost).\n"
        )
        buf.write(
            "  Savings: no AdamW on decoder params only; step time change is negligible "
            "vs full step.\n"
        )
        if "forward_total" in phase_us:
            dec_module_us = sum(
                us for us, key, _ in module_rows if key == "base_model.decoder"
            )
            if dec_module_us > 0:
                pct = 100.0 * dec_module_us / phase_us["forward_total"]
                buf.write(
                    f"  This run: decoder forward ≈ {dec_module_us/1000:.1f} ms "
                    f"({pct:.0f}% of forward_total).\n"
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="ckpt/AdaPoinTr_ps55.pth")
    ap.add_argument("--tag", default="profile")
    ap.add_argument("--batch-size", type=int, default=16, help="per-GPU batch size")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--profile-steps", type=int, default=2)
    ap.add_argument("--out-dir", default="logs/complete")
    ap.add_argument(
        "--forward-only",
        action="store_true",
        help="profile forward+loss only (no backward/optimizer)",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for profiler")

    cfg = cfg_from_yaml_file(args.config)
    per_gpu_bs = args.batch_size
    cfg.dataset.train.others.bs = per_gpu_bs

    class _MiniArgs:
        launcher = "none"
        distributed = False
        local_rank = 0
        num_workers = 2

    mini = _MiniArgs()
    _, train_loader = builder.dataset_builder(mini, cfg.dataset.train)

    model = builder.model_builder(cfg.model).cuda().train()
    if args.ckpt:
        builder.load_model(model, args.ckpt, logger=None)
    model = nn.DataParallel(model)
    optimizer = builder.build_optimizer(model, cfg)
    module_hooks = _install_module_hooks(model.module, fine_grained=True)

    batch = next(iter(train_loader))
    taxonomy_ids, model_ids, data = batch
    partial = data[0].cuda(non_blocking=True)
    gt = data[1].cuda(non_blocking=True)

    step_fn = lambda: _one_step(
        model, partial, gt, optimizer, forward_only=args.forward_only
    )
    for _ in range(args.warmup):
        step_fn()
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=False,
    ) as prof:
        for _ in range(args.profile_steps):
            step_fn()
        torch.cuda.synchronize()

    for h in module_hooks:
        h.remove()

    buf = io.StringIO()
    mode = "forward_only" if args.forward_only else "full_train_step"
    buf.write(f"=== torch.profiler: {args.tag} ({mode}) ===\n")
    buf.write(f"config: {args.config}\n")
    buf.write(f"ckpt: {args.ckpt}\n")
    buf.write(f"batch_size(per GPU): {per_gpu_bs}\n")
    buf.write(f"profile_steps: {args.profile_steps}\n")
    buf.write(f"mode: {mode}\n\n")

    buf.write("--- Top ops by cuda_time_total (us) ---\n")
    buf.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=35))
    buf.write("\n\n--- Top ops by self_cuda_time_total (us) ---\n")
    buf.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
    _write_module_breakdown(buf, prof, args.forward_only)

    keyword_map = {
        "chamfer_cd": ["chamfer", "ChamferDistance"],
        "matmul_gemm": ["mm", "addmm", "bmm", "matmul", "Linear"],
        "attention_softmax": ["softmax", "Attention", "attn"],
        "normalization": ["layer_norm", "batch_norm", "LayerNorm", "BatchNorm", "native_layer_norm"],
        "msf_sort_index": ["sort", "index", "gather", "scatter"],
        "conv1d_rebuild": ["conv1d", "Conv1d"],
        "optimizer_adamw": ["adam", "AdamW", "foreach"],
        "memcpy_sync": ["memcpy", "synchronize", "clone", "copy_"],
    }
    agg = _aggregate_keywords(prof, keyword_map)
    buf.write("\n--- Keyword buckets (cuda_time_total) ---\n")
    for name, (us, pct) in agg.items():
        buf.write(f"  {name:22s}  {us/1000:8.2f} ms  ({pct:5.1f}%)\n")

    os.makedirs(args.out_dir, exist_ok=True)
    out_txt = os.path.join(args.out_dir, f"profile_{args.tag}.txt")
    out_json = os.path.join(args.out_dir, f"profile_{args.tag}.json")
    text = buf.getvalue()
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    prof.export_chrome_trace(out_json.replace(".json", ".chrome.json"))
    print(text)
    print(f"Saved: {out_txt}")
    print(f"Chrome trace: {out_json.replace('.json', '.chrome.json')}")


if __name__ == "__main__":
    main()
