"""同一 GT 在侧视 vs 斜视下的对比（说明为何之前像长方形）。"""
import os
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gt = np.asarray(
    o3d.io.read_point_cloud(
        os.path.join(ROOT, "data/PCN/test/complete/03001627/91b8fe4616208bd4cf752e9bed38184f.pcd")
    ).points
)
c = gt.mean(0)
s = (gt.max(0) - gt.min(0)).max() * 0.55
lim = (c - s, c + s)

views = [
    ("侧视 (10°, 90°) — 易变成长方形", 10, 90),
    ("斜视 (22°, -58°) — 推荐", 22, -58),
    ("stage1 (18°, 45°)", 18, 45),
]

fig = plt.figure(figsize=(15, 4.5), facecolor="white")
for i, (title, elev, azim) in enumerate(views):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c="#E8A020", s=1.5, alpha=0.9, depthshade=False)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(lim[0][0], lim[1][0])
    ax.set_ylim(lim[0][1], lim[1][1])
    ax.set_zlim(lim[0][2], lim[1][2])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, fontweight="bold")

out = os.path.join(os.path.dirname(__file__), "output", "view_compare_chair_gt.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(out)
