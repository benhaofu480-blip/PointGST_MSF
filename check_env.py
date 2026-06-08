"""环境检测脚本 - 在远程服务器上运行: python check_env.py"""
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

deps = [
    ("timm", "import timm"),
    ("knn_cuda", "import knn_cuda"),
    ("pointnet2_ops", "import pointnet2_ops"),
    ("chamfer_dist", "from extensions.chamfer_dist import ChamferDistanceL1"),
    ("easydict", "import easydict"),
    ("tensorboardX", "import tensorboardX"),
    ("pyyaml", "import yaml"),
    ("numpy", "import numpy"),
    ("PGST module", "from models.PGST import PCSA"),
    ("z_order", "from models.z_order import xyz2key"),
    ("PartSeg model", "from models.PointTransformerPartSeg_PGST import PointTransformerPartSeg_PGST"),
    ("utils.misc", "from utils import misc"),
]

print("\n--- Dependency Check ---")
for name, stmt in deps:
    try:
        exec(stmt)
        print(f"  [OK] {name}")
    except Exception as e:
        print(f"  [MISSING] {name} - {e}")

print("\n--- Dataset Check ---")
import os
sn = "data/ShapeNetPart"
if os.path.isdir(sn):
    files = ["train_points.npy", "train_labels.npy", "train_seg.npy",
             "test_points.npy", "test_labels.npy", "test_seg.npy"]
    for f in files:
        p = os.path.join(sn, f)
        if os.path.exists(p):
            import numpy as np
            a = np.load(p, allow_pickle=True)
            print(f"  [OK] {f} shape={a.shape}")
        else:
            print(f"  [MISSING] {f}")
else:
    print(f"  [MISSING] {sn}/ directory")

print("\n--- Pretrained Check ---")
ckpt = "pretrained/pointbert_pretrain.pth"
if os.path.exists(ckpt):
    print(f"  [OK] {ckpt}")
else:
    print(f"  [MISSING] {ckpt}")
