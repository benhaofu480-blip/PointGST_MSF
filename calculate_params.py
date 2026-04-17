#!/usr/bin/env python3
"""
计算Point-BERT + PointGST的准确参数量
"""

import torch
import torch.nn as nn

# 配置
config = {
    'trans_dim': 384,
    'depth': 12,
    'num_heads': 6,
    'group_size': 32,
    'num_group': 64,
    'encoder_dims': 384,
    'rank': 36,
    'NUM_PARTS': 50
}

# 1. Encoder
def count_encoder():
    # first_conv: 3->128->256
    # second_conv: 512->512->384
    params = 0
    params += (3 * 128 + 128)  # conv1 weight + bias
    params += (128 * 256 + 256)  # conv2 weight + bias
    params += (512 * 512 + 512)  # second_conv1
    params += (512 * 384 + 384)  # second_conv2
    return params

# 2. PCSA adapter
def count_pcsa():
    dim = config['trans_dim']
    rank = config['rank']
    
    params = 0
    params += dim * rank  # down
    params += rank * dim  # up
    params += rank * rank  # adapt1
    params += rank * 2    # norm1
    params += rank * 2    # norm2
    return params

# 3. Transformer Block
def count_block():
    dim = config['trans_dim']
    heads = config['num_heads']
    
    params = 0
    # MultiheadAttention: qkv + out_proj
    params += dim * dim * 3  # qkv
    params += dim * dim      # out_proj
    # MLP
    hidden = dim * 4
    params += dim * hidden   # fc1
    params += hidden * dim   # fc2
    # LayerNorms
    params += dim * 2 * 2    # norm1 + norm2
    # PCSA
    params += count_pcsa()
    
    return params

# 4. Transformer Encoder (12层)
def count_transformer():
    block_params = count_block()
    return block_params * config['depth']

# 5. cls token & pos embed
def count_token():
    dim = config['trans_dim']
    # cls_token
    params = dim
    # cls_pos
    params += dim
    # pos_embed network: 3->128->dim
    params += 3 * 128 + 128
    params += 128 * dim + dim
    return params

# 6. 上采样网络
def count_upsample():
    dim = config['trans_dim']
    params = 0
    params += dim * 512 + 512  # conv1
    params += 512 * 256 + 256  # conv2
    params += 256 * 128 + 128  # conv3
    params += 128 * dim + dim  # conv4
    return params

# 7. 分割头
def count_seg_head():
    dim = config['trans_dim']
    n_parts = config['NUM_PARTS']
    params = 0
    params += dim * 256 + 256  # conv1
    params += 256 * 128 + 128  # conv2
    params += 128 * n_parts + n_parts  # conv3
    return params

# 8. 总计
def count_total():
    encoder = count_encoder()
    transformer = count_transformer()
    token = count_token()
    upsample = count_upsample()
    seg_head = count_seg_head()
    
    total = encoder + transformer + token + upsample + seg_head
    
    # PCSA适配器（12层）
    pcsa = count_pcsa() * config['depth']
    
    # 可训练参数（PCSA + 分割头 + 上采样）
    trainable = pcsa + seg_head + upsample
    
    return total, trainable

if __name__ == '__main__':
    total, trainable = count_total()
    
    print("="*60)
    print("Point-BERT + PointGST 参数量计算")
    print("="*60)
    print(f"Total Parameters: {total/1e6:.2f}M")
    print(f"Trainable Parameters: {trainable/1e6:.2f}M")
    print(f"Frozen Parameters: {(total-trainable)/1e6:.2f}M")
    print(f"Trainable Ratio: {trainable/total*100:.2f}%")
    print("="*60)
    print("\nComponents:")
    print(f"  Encoder: {count_encoder()/1e6:.2f}M")
    print(f"  Transformer (12 layers): {count_transformer()/1e6:.2f}M")
    print(f"  PCSA (12 layers): {count_pcsa()*config['depth']/1e6:.2f}M")
    print(f"  Token & PosEmbed: {count_token()/1e6:.2f}M")
    print(f"  Upsample: {count_upsample()/1e6:.2f}M")
    print(f"  SegHead: {count_seg_head()/1e6:.2f}M")
