"""
快速查看点云文件 - 服务器版本（保存为图片）
用法: python view_point_cloud.py <文件名>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无图形界面模式
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def save_point_cloud_image(filename, output_dir='./visualize_output'):
    """保存点云为图片文件"""
    points = np.loadtxt(filename)
    
    # 创建3个视角的图片
    fig = plt.figure(figsize=(18, 5))
    
    # 视角1: 3D斜视
    ax1 = fig.add_subplot(131, projection='3d')
    colors = points[:, 2]  # Z坐标着色
    scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                         c=colors, cmap='viridis', s=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View')
    
    # 视角2: XY平面（俯视图）
    ax2 = fig.add_subplot(132)
    ax2.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='viridis', s=0.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top View (XY)')
    ax2.set_aspect('equal')
    
    # 视角3: XZ平面（侧视图）
    ax3 = fig.add_subplot(133)
    ax3.scatter(points[:, 0], points[:, 2], c=points[:, 2], cmap='viridis', s=0.5)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('Side View (XZ)')
    ax3.set_aspect('equal')
    
    # 提取文件名作为标题
    basename = filename.split('/')[-1]
    fig.suptitle(f'{basename}\n{len(points)} points', fontsize=12)
    
    plt.tight_layout()
    
    # 保存为png
    output_file = filename.replace('.xyz', '.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"已保存: {output_file}")
    return output_file

def compare_and_save(file1, file2, output_dir='./visualize_output'):
    """对比两个点云并保存为图片"""
    points1 = np.loadtxt(file1)
    points2 = np.loadtxt(file2)
    
    print(f"{file1}: {len(points1)} points")
    print(f"{file2}: {len(points2)} points")
    
    fig = plt.figure(figsize=(18, 10))
    
    # 文件1 - 3D
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c='blue', s=0.5)
    ax1.set_title(f'GT: {file1.split("/")[-1]}')
    
    # 文件1 - XY
    ax2 = fig.add_subplot(232)
    ax2.scatter(points1[:, 0], points1[:, 1], c='blue', s=0.5)
    ax2.set_title('GT Top View')
    ax2.set_aspect('equal')
    
    # 文件1 - XZ
    ax3 = fig.add_subplot(233)
    ax3.scatter(points1[:, 0], points1[:, 2], c='blue', s=0.5)
    ax3.set_title('GT Side View')
    ax3.set_aspect('equal')
    
    # 文件2 - 3D
    ax4 = fig.add_subplot(234, projection='3d')
    ax4.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='red', s=0.5)
    ax4.set_title(f'Pred: {file2.split("/")[-1]}')
    
    # 文件2 - XY
    ax5 = fig.add_subplot(235)
    ax5.scatter(points2[:, 0], points2[:, 1], c='red', s=0.5)
    ax5.set_title('Pred Top View')
    ax5.set_aspect('equal')
    
    # 文件2 - XZ
    ax6 = fig.add_subplot(236)
    ax6.scatter(points2[:, 0], points2[:, 2], c='red', s=0.5)
    ax6.set_title('Pred Side View')
    ax6.set_aspect('equal')
    
    fig.suptitle('Blue=GT, Red=Prediction', fontsize=14)
    plt.tight_layout()
    
    # 保存
    name1 = file1.split('/')[-1].replace('.xyz', '')
    name2 = file2.split('/')[-1].replace('.xyz', '')
    output_file = f"{output_dir}/compare_{name1}_vs_{name2}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"已保存对比图: {output_file}")
    return output_file

if __name__ == '__main__':
    import os
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  查看单个文件: python view_point_cloud.py sofa_0_gt.xyz")
        print("  对比两个文件: python view_point_cloud.py sofa_0_gt.xyz sofa_0_fine.xyz")
        print("\n当前目录下的可视化文件:")
        if os.path.exists('./visualize_output'):
            files = sorted([f for f in os.listdir('./visualize_output') if f.endswith('.xyz')])
            for f in files[:10]:
                print(f"  - visualize_output/{f}")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        save_point_cloud_image(sys.argv[1])
    else:
        compare_and_save(sys.argv[1], sys.argv[2])
