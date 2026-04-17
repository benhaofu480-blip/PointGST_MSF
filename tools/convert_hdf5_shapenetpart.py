"""
ShapeNetPart HDF5 -> npy 转换脚本
适配 PointGST 训练所需的格式

支持两种常见HDF5格式：
  格式A (PointCloudDatasets等): {'points': (N, 2048, 6), 'normals': ..., 'labels': (N,), 'seg_labels': (N, 2048)}
  格式B (常见): {'data': (N, 2048, 3/6), 'label': ..., 'seg': ...}
"""
import h5py
import numpy as np
import os
import sys


def inspect_hdf5(path):
    """打印HDF5文件结构"""
    print(f"\n=== Inspecting {path} ===")
    with h5py.File(path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"  Group: {name}")
        f.visititems(print_structure)
    print()


def convert_formatA(h5path, out_dir):
    """
    格式A: datasets可能在顶层或某个group下
    key映射:
      points/xyz/verts -> (N, 2048, 3) 取前3维
      labels/label/category -> (N,) 物体类别
      seg_labels/seg/part_seg -> (N, 2048) 部件标签
    """
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(h5path, 'r') as f:
        # 尝试各种可能的key名
        def find_key(f, candidates):
            for c in candidates:
                if c in f:
                    return f[c][:]
            return None

        for split_name, is_train in [('train', True), ('test', False)]:
            # 尝试 train/test 分组或 split标记
            split_group = None
            for candidate in [split_name, split_name + '_data', split_name.capitalize()]:
                if candidate in f:
                    split_group = f[candidate]
                    break

            src = split_group if split_group is not None else f

            points = find_key(src, ['points', 'xyz', 'verts', 'point_clouds', 'data'])
            labels = find_key(src, ['labels', 'label', 'category', 'cat_id'])
            seg = find_key(src, ['seg_labels', 'seg', 'part_seg', 'seg_labels', 'pid'])

            if points is None or labels is None or seg is None:
                print(f"  [SKIP] {split_name}: 无法找到必要的数据key")
                print(f"    尝试过的key: points={points is not None}, labels={labels is not None}, seg={seg is not None}")
                continue

            # 只取前3维 (xyz)
            if points.ndim == 3 and points.shape[-1] > 3:
                points = points[:, :, :3]

            N = points.shape[0]
            print(f"  {split_name}: N={N}, points={points.shape}, labels={labels.shape}, seg={seg.shape}")

            # 重新采样到2048点（如果不是）
            npoints = 2048
            if points.shape[1] != npoints:
                print(f"  Resampling from {points.shape[1]} to {npoints} points...")
                new_points = np.zeros((N, npoints, 3), dtype=np.float32)
                new_seg = np.zeros((N, npoints), dtype=np.int64)
                for i in range(N):
                    idx = np.random.choice(points.shape[1], npoints, replace=points.shape[1] < npoints)
                    new_points[i] = points[i, idx]
                    new_seg[i] = seg[i, idx]
                points = new_points
                seg = new_seg

            # 确保seg是int64
            seg = seg.astype(np.int64)
            labels = labels.astype(np.int64).reshape(-1)

            # ShapeNetPart标准: 16个类别, label 0-15
            # 如果label从1开始，需要减1
            if labels.min() > 0:
                # 检查是否是原始synset id还是index
                unique_labels = np.unique(labels)
                if len(unique_labels) == 16:
                    print(f"  Adjusting labels: min={labels.min()}, mapping to 0-15")
                    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
                    labels = np.array([label_map[l] for l in labels])

            # 部件标签重映射：将不连续的部件标签映射为0起始连续
            # 按类别分别处理
            class2parts_standard = {
                0: [0, 1, 2, 3], 1: [4, 5], 2: [6, 7], 3: [8, 9, 10, 11],
                4: [12, 13, 14, 15], 5: [16, 17, 18], 6: [19, 20, 21], 7: [22, 23],
                8: [24, 25, 26, 27], 9: [28, 29, 30, 31], 10: [32, 33, 34],
                11: [35, 36, 37], 12: [38, 39, 40], 13: [41, 42, 43],
                14: [44, 45, 46], 15: [47, 48, 49]
            }

            # 检查部件标签是否已经是标准格式
            unique_parts = np.unique(seg)
            max_part = unique_parts.max()
            print(f"  Unique parts: {len(unique_parts)}, range: [{unique_parts.min()}, {max_part}]")

            if max_part >= 50 or len(unique_parts) > 50:
                print(f"  Remapping seg labels to standard format (0-49)...")
                seg_new = np.zeros_like(seg)
                for cls_id in range(16):
                    mask = labels == cls_id
                    if not mask.any():
                        continue
                    cls_parts = seg[mask]
                    unique_cls_parts = sorted(np.unique(cls_parts).tolist())
                    part_map = {old: new for new, old in enumerate(unique_cls_parts)}
                    expected_parts = class2parts_standard[cls_id]
                    if len(unique_cls_parts) != len(expected_parts):
                        print(f"    WARNING: Class {cls_id} has {len(unique_cls_parts)} parts, expected {len(expected_parts)}")
                    for old, new in part_map.items():
                        seg_new[mask & (seg == old)] = new + expected_parts[0]
                seg = seg_new

            np.save(os.path.join(out_dir, f'{split_name}_points.npy'), points.astype(np.float32))
            np.save(os.path.join(out_dir, f'{split_name}_labels.npy'), labels)
            np.save(os.path.join(out_dir, f'{split_name}_seg.npy'), seg)
            print(f"  Saved to {out_dir}/{split_name}_*.npy")

    print("\nDone!")


def convert_formatB(h5path, out_dir):
    """
    格式B: 所有数据在顶层, 没有train/test分组
    需要外部提供train/test split
    """
    print("数据没有train/test分组，需要手动指定split文件")
    print("建议改用官方数据集")
    sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_hdf5_shapenetpart.py <hdf5_file> [output_dir]")
        print("\nExamples:")
        print("  # 先检查HDF5结构")
        print("  python convert_hdf5_shapenetpart.py data/ShapeNetPart/ply_hdf5_2048/ply_data_train.h5")
        sys.exit(0)

    h5path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/ShapeNetPart'

    if not os.path.exists(h5path):
        print(f"File not found: {h5path}")
        sys.exit(1)

    # 先检查结构
    inspect_hdf5(h5path)

    # 转换
    convert_formatA(h5path, out_dir)
