"""Side-by-side comparison of v2 baseline vs Flow Matching v1 coarse eval images."""
import os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

V2_DIR  = "./visualize_output/coarse_eval_v2"
FM_DIR  = "./visualize_output/coarse_eval_fm_v1"
OUT_DIR = "./visualize_output/comparison_v2_vs_fm"

os.makedirs(OUT_DIR, exist_ok=True)

categories = ["airplane", "cabinet", "car", "chair", "lamp", "sofa", "table", "watercraft"]

for suffix in ["coarse_eval", "multiangle", "offset_hist"]:
    fig, axes = plt.subplots(len(categories), 2, figsize=(40, 6 * len(categories)))

    for i, cat in enumerate(categories):
        v2_files = sorted(glob.glob(os.path.join(V2_DIR, f"{cat}_*_{suffix}.png")))
        fm_files = sorted(glob.glob(os.path.join(FM_DIR, f"{cat}_*_{suffix}.png")))

        for j, (files, label) in enumerate([(v2_files, "v2 baseline"), (fm_files, "FM v1")]):
            ax = axes[i, j]
            if files:
                img = mpimg.imread(files[0])
                ax.imshow(img)
                ax.set_title(f"{cat.upper()} — {label}", fontsize=14, fontweight="bold")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=20)
                ax.set_title(f"{cat.upper()} — {label}", fontsize=14)
            ax.axis("off")

    plt.suptitle(f"v2 Baseline vs Flow Matching v1 — {suffix.replace('_', ' ').title()}",
                 fontsize=18, fontweight="bold", y=1.0)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"compare_{suffix}.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

# Summary comparison table
v2_summary = open(os.path.join(V2_DIR, "summary.txt")).readlines()
fm_summary = open(os.path.join(FM_DIR, "summary.txt")).readlines()

with open(os.path.join(OUT_DIR, "comparison_summary.txt"), "w") as f:
    f.write("=" * 120 + "\n")
    f.write("COMPARISON: v2 Baseline vs Flow Matching v1 (Coarse Point Evaluation)\n")
    f.write("=" * 120 + "\n\n")
    f.write("--- v2 Baseline ---\n")
    for line in v2_summary:
        f.write(line)
    f.write("\n--- Flow Matching v1 ---\n")
    for line in fm_summary:
        f.write(line)
    f.write("\n" + "=" * 120 + "\n")
    f.write("Per-category CDL1 (overall, fine level):\n")
    f.write(f"  v2 baseline best (E150): CDL1=8.097  F-Score=0.774\n")
    f.write(f"  FM v1 best      (E90):   CDL1=8.093  F-Score=0.764\n")
    f.write("=" * 120 + "\n")

print(f"\nAll comparison outputs saved to: {os.path.abspath(OUT_DIR)}")
