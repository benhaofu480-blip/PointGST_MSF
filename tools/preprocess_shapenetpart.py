import os
import numpy as np
import json

def load_shapenetpart_data(root, split='train'):
    cat_file = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    with open(cat_file, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]  # cat['Airplane'] = '02691156'

    synset_ids = sorted(cat.values())
    class2index = {synset_id: i for i, synset_id in enumerate(synset_ids)}

    # 目标标签范围：每个类别映射到 0-49 的全局唯一区间
    class2target = {
        '02691156': [0, 1, 2, 3], '02773838': [4, 5], '02954340': [6, 7],
        '02958343': [8, 9, 10, 11], '03001627': [12, 13, 14, 15],
        '03261776': [16, 17, 18], '03467517': [19, 20, 21],
        '03624134': [22, 23], '03636649': [24, 25, 26, 27],
        '03642806': [28, 29, 30, 31], '03790512': [32, 33, 34],
        '03797390': [35, 36, 37], '03948459': [38, 39, 40],
        '04099429': [41, 42, 43], '04225987': [44, 45, 46],
        '04379243': [47, 48, 49]
    }

    # 读取split JSON
    split_file_map = {
        'train': 'shuffled_train_file_list.json',
        'val':   'shuffled_val_file_list.json',
        'test':  'shuffled_test_file_list.json',
    }
    split_file = os.path.join(root, 'train_test_split', split_file_map[split])

    with open(split_file, 'r') as f:
        lines = json.load(f)

    ids = []
    for synset_id in synset_ids:
        for line in lines:
            line = line.strip()
            if synset_id in line:
                parts = line.replace('shape_data/', '').split('/')
                ids.append((parts[0], parts[1]))

    all_points = []
    all_labels = []
    all_seg = []

    for cls, file_id in ids:
        data = np.loadtxt(os.path.join(root, cls, file_id + '.txt')).astype(np.float32)

        points = data[:, :3]
        seg = data[:, -1].astype(np.int64)

        # 动态检测该类别的原始标签（排除0=背景）
        valid_original = sorted(set(s for s in np.unique(seg) if s > 0))
        target = class2target[cls]

        # 正确映射：原始标签 → 全局唯一标签
        # 若该shape缺少某些部件（正常现象），按顺序截取target
        n_parts = min(len(valid_original), len(target))
        part_mapping = {old: new for old, new in zip(valid_original[:n_parts], target[:n_parts])}
        seg = np.array([part_mapping.get(s, 0) for s in seg])

        if len(points) >= 2048:
            choice = np.random.choice(len(points), 2048, replace=False)
        else:
            choice = np.random.choice(len(points), 2048, replace=True)

        points = points[choice]
        seg = seg[choice]

        label = class2index[cls]

        all_points.append(points)
        all_labels.append(label)
        all_seg.append(seg)

    return np.array(all_points), np.array(all_labels), np.array(all_seg)

if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
    out_dir = 'data/ShapeNetPart'
    os.makedirs(out_dir, exist_ok=True)

    for split in ['train', 'val', 'test']:
        print(f"Processing {split} set...")
        points, labels, seg = load_shapenetpart_data(root, split)
        np.save(os.path.join(out_dir, f'{split}_points.npy'), points)
        np.save(os.path.join(out_dir, f'{split}_labels.npy'), labels)
        np.save(os.path.join(out_dir, f'{split}_seg.npy'), seg)

        print(f"  seg range: [{seg.min()}, {seg.max()}]")
        for c in range(16):
            mask = labels == c
            if mask.any():
                parts = np.unique(seg[mask])
                print(f"    Category {c}: parts={sorted(parts.tolist())}, n={mask.sum()}")
        print(f"  Total: {len(points)} samples\n")

    print("Done!")
