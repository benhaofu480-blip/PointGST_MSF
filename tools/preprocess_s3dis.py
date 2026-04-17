"""
S3DIS 数据预处理脚本
将原始 S3DIS 房间点云切割成固定大小的 block，保存为 npy 格式

用法: python tools/preprocess_s3dis.py --data_root data/stanford3d --output_root data/S3DIS

输出格式:
  data/S3DIS/
    Area_1_block_1.0_0.5.npy  (每个block: dict with keys 'xyz', 'rgb', 'labels', 'room_name')
    Area_1_block_1.0_0.5_index.txt
    ...
"""
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm


# S3DIS 标准类别映射 (原始标注可能有不连续的label编号)
# 共13个有效类别: ceiling, floor, wall, beam, column, window, door,
#                chair, table, bookcase, sofa, board, clutter
CLASS_NAMES = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
    'chair', 'table', 'bookcase', 'sofa', 'board', 'clutter'
]
# 有些原始标注用 'stairs' (label=14) 但标准benchmark不包含它
VALID_CLASSES = set(range(13))  # 0-12


def collect_room_point_clouds(data_root, area_id):
    """收集一个Area下所有房间的点云"""
    area_path = os.path.join(data_root, f'Area_{area_id}')
    if not os.path.exists(area_path):
        print(f'[WARNING] Area_{area_id} not found at {area_path}, skipping')
        return []

    rooms = []
    # 搜索所有 .txt 文件
    txt_files = sorted(glob.glob(os.path.join(area_path, '*', '*.txt')))
    if not txt_files:
        print(f'[WARNING] No .txt files found in {area_path}')
        return []

    for txt_file in txt_files:
        room_name = os.path.splitext(os.path.basename(txt_file))[0]
        try:
            data = np.loadtxt(txt_file, dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            # 原始格式: x y z r g b label
            # 有些文件是 6 列 (xyz+rgb, 无label), 有些是 7 列
            if data.shape[1] >= 7:
                xyz = data[:, :3]
                rgb = data[:, 3:6]
                labels = data[:, 6].astype(np.int32)
            elif data.shape[1] == 6:
                xyz = data[:, :3]
                rgb = data[:, 3:6]
                labels = np.zeros(xyz.shape[0], dtype=np.int32) - 1  # 无标注
            else:
                print(f'[WARNING] Unexpected shape {data.shape} in {txt_file}')
                continue

            # 将不在有效类别中的点标记为 -1 (忽略)
            valid_mask = np.isin(labels, list(VALID_CLASSES))
            labels[~valid_mask] = -1

            rooms.append({
                'room_name': room_name,
                'area_id': area_id,
                'xyz': xyz,
                'rgb': rgb,
                'labels': labels,
            })
        except Exception as e:
            print(f'[ERROR] Failed to load {txt_file}: {e}')

    print(f'Area_{area_id}: loaded {len(rooms)} rooms')
    return rooms


def room_to_blocks(room, block_size=1.0, stride=0.5, num_points=4096):
    """将房间点云切割成固定大小的block"""
    xyz = room['xyz']
    rgb = room['rgb']
    labels = room['labels']

    # 计算房间的边界
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)

    blocks = []
    # 沿 x, y 方向滑动窗口 (S3DIS是地面在XY平面)
    x_ranges = np.arange(xyz_min[0], xyz_max[0] - block_size + 1e-6, stride)
    y_ranges = np.arange(xyz_min[1], xyz_max[1] - block_size + 1e-6, stride)

    for x_start in x_ranges:
        for y_start in y_ranges:
            x_end = x_start + block_size
            y_end = y_start + block_size

            # 选取block内的点
            mask = (
                (xyz[:, 0] >= x_start) & (xyz[:, 0] < x_end) &
                (xyz[:, 1] >= y_start) & (xyz[:, 1] < y_end)
            )
            block_xyz = xyz[mask]
            block_rgb = rgb[mask]
            block_labels = labels[mask]

            if block_xyz.shape[0] < 100:
                continue  # 点太少，跳过

            # 采样到固定数量
            if block_xyz.shape[0] >= num_points:
                choice = np.random.choice(block_xyz.shape[0], num_points, replace=False)
            else:
                choice = np.random.choice(block_xyz.shape[0], num_points, replace=True)

            blocks.append({
                'xyz': block_xyz[choice],
                'rgb': block_rgb[choice],
                'labels': block_labels[choice],
                'room_name': room['room_name'],
                'area_id': room['area_id'],
            })

    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/stanford3d',
                        help='S3DIS原始数据根目录 (包含 Area_1 ~ Area_6)')
    parser.add_argument('--output_root', default='data/S3DIS',
                        help='预处理输出目录')
    parser.add_argument('--block_size', type=float, default=1.0)
    parser.add_argument('--stride', type=float, default=0.5)
    parser.add_argument('--num_points', type=int, default=4096)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    # 标准5-fold: 测试 Area_5, 训练 Area_1,2,3,4,6
    test_area = 5
    train_areas = [1, 2, 3, 4, 6]

    for split_name, areas in [('train', train_areas), ('test', [test_area])]:
        all_blocks = []
        print(f'\n=== Processing {split_name} (Areas: {areas}) ===')
        for area_id in areas:
            rooms = collect_room_point_clouds(args.data_root, area_id)
            for room in tqdm(rooms, desc=f'Area_{area_id} blocks'):
                blocks = room_to_blocks(
                    room,
                    block_size=args.block_size,
                    stride=args.stride,
                    num_points=args.num_points,
                )
                all_blocks.extend(blocks)

        print(f'{split_name}: total {len(all_blocks)} blocks')

        # 保存为 npy
        output_path = os.path.join(args.output_root,
                                   f'{split_name}_block_{args.block_size}_{args.stride}.npy')
        data_list = []
        for b in all_blocks:
            data_list.append({
                'xyz': b['xyz'].astype(np.float32),
                'rgb': b['rgb'].astype(np.float32),
                'labels': b['labels'].astype(np.int64),
                'room_name': b['room_name'],
                'area_id': b['area_id'],
            })
        np.save(output_path, data_list)
        print(f'Saved to {output_path}')

        # 保存索引文件 (room_name → block indices) 用于房间级评估
        index_path = os.path.join(args.output_root,
                                  f'{split_name}_block_{args.block_size}_{args.stride}_index.txt')
        with open(index_path, 'w') as f:
            for i, b in enumerate(all_blocks):
                f.write(f'{i}\tArea_{b["area_id"]}/{b["room_name"]}\n')
        print(f'Saved index to {index_path}')


if __name__ == '__main__':
    main()
