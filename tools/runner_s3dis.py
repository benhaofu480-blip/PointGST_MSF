"""
S3DIS 语义分割训练/测试函数
评估协议: 13类 mIoU (忽略 label=-1 的点)
"""
import os
import time
import torch
import torch.nn as nn
import numpy as np

from tools import builder
from utils import misc, dist_utils
from utils.logger import *
from utils.AverageMeter import AverageMeter
from pointnet2_ops import pointnet2_utils


class IoU_Metric:
    """S3DIS IoU 指标"""
    def __init__(self, num_classes=13):
        self.num_classes = num_classes
        self.acc = 0.
        self.class_iou = np.zeros(num_classes)
        self.class_count = np.zeros(num_classes)

    def update(self, pred, label):
        """更新单个 batch 的 IoU 统计"""
        pred = pred.cpu().numpy()
        label = label.cpu().numpy()
        for i in range(self.num_classes):
            intersection = np.sum((pred == i) & (label == i))
            union = np.sum((pred == i) | (label == i))
            if union > 0:
                self.class_iou[i] += intersection / union
                self.class_count[i] += 1

    def compute(self):
        """计算最终 mIoU"""
        valid = self.class_count > 0
        if valid.sum() == 0:
            return 0.0, np.zeros(self.num_classes)
        miou = np.mean(self.class_iou[valid] / self.class_count[valid]) * 100
        per_class = self.class_iou / (self.class_count + 1e-8) * 100
        return miou, per_class

    def better_than(self, other):
        self_miou, _ = self.compute()
        other_miou, _ = other.compute()
        return self_miou > other_miou

    def state_dict(self):
        miou, _ = self.compute()
        return {'acc': miou}

    def __str__(self):
        miou, per_class = self.compute()
        return f'mIoU={miou:.2f}%'


S3DIS_CLASS_NAMES = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
    'chair', 'table', 'bookcase', 'sofa', 'board', 'clutter'
]


def run_net_s3dis(args, config, train_writer=None, val_writer=None, logger=None):
    """S3DIS 语义分割训练"""
    if logger is None:
        logger = get_logger(args.log_name)

    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = \
        builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)

    # build model
    base_model = builder.model_builder(config.model)

    # parameter setting
    start_epoch = 0
    best_metrics = IoU_Metric(config.model.NUM_CLASSES)
    metrics = IoU_Metric(config.model.NUM_CLASSES)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = IoU_Metric(config.model.NUM_CLASSES)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    if args.distributed:
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    base_model.zero_grad()
    misc.summary_parameters(base_model, logger=logger)

    # 冻结 backbone BN
    for name, module in base_model.module.named_modules():
        if any(p in name for p in ['propagation_', 'dgcnn_pro', 'conv1', 'bn1', 'drop1', 'conv2']):
            continue
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        losses = AverageMeter(['loss'])
        num_iter = 0
        n_batches = len(train_dataloader)

        npoints = config.npoints
        for idx, data in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            batch_start_time = time.time()

            points = data['points'].cuda()   # (B, N, 6)
            seg = data['seg'].cuda()         # (B, N)

            # NaN 诊断（前3个batch检测数据质量）
            if idx < 3 and (torch.isnan(points).any() or torch.isinf(points).any()):
                nan_count = torch.isnan(points).sum().item()
                inf_count = torch.isinf(points).sum().item()
                print_log(f'[WARN] Batch {idx}: points has {nan_count} NaN, {inf_count} Inf', logger=logger)

            # FPS 采样 (同步采样 seg 标签)
            if points.size(1) > npoints:
                fps_idx = pointnet2_utils.furthest_point_sample(points[:, :, :3], npoints)
                points = pointnet2_utils.gather_operation(
                    points.transpose(1, 2).contiguous(), fps_idx
                ).transpose(1, 2).contiguous()
                fps_idx_expand = fps_idx.expand(-1, -1, seg.shape[-1])
                seg = torch.gather(seg.unsqueeze(1), 1, fps_idx_expand).squeeze(1)

            ret = base_model(points)
            loss = base_model.module.loss_ce(ret.reshape(-1, ret.shape[-1]), seg.reshape(-1))

            loss.backward()

            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            losses.update([loss.item()])

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            metrics = validate_s3dis(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
            better = metrics.better_than(best_metrics)

            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                        logger=logger)
                print_log("--------------------------------------------------------------------------------------------",
                          logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate_s3dis(base_model, test_dataloader, epoch, val_writer, args, config, logger=None):
    """S3DIS 验证"""
    metric = IoU_Metric(config.model.NUM_CLASSES)
    num_classes = config.model.NUM_CLASSES
    npoints = config.npoints

    base_model.eval()
    test_loss_sum = 0.0
    test_loss_count = 0

    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            points = data['points'].cuda()
            seg = data['seg'].cuda()

            if points.size(1) > npoints:
                fps_idx = pointnet2_utils.furthest_point_sample(points[:, :, :3], npoints)
                points = pointnet2_utils.gather_operation(
                    points.transpose(1, 2).contiguous(), fps_idx
                ).transpose(1, 2).contiguous()
                fps_idx_expand = fps_idx.expand(-1, -1, seg.shape[-1])
                seg = torch.gather(seg.unsqueeze(1), 1, fps_idx_expand).squeeze(1)

            ret = base_model(points)
            loss = base_model.module.loss_ce(ret.reshape(-1, ret.shape[-1]), seg.reshape(-1))
            test_loss_sum += loss.item()
            test_loss_count += 1

            pred = ret.argmax(-1)
            metric.update(pred, seg)

            if idx % 20 == 0:
                print_log('[Validation] EPOCH: %d BATCH: %d/%d' % (epoch, idx, len(test_dataloader)), logger=logger)

    avg_loss = test_loss_sum / max(test_loss_count, 1)
    miou, per_class_iou = metric.compute()

    print_log('============================================================', logger=logger)
    print_log('[Validation] EPOCH: %d' % epoch, logger=logger)
    print_log('  Loss: %.4f' % avg_loss, logger=logger)
    print_log('  mIoU: %.2f%%' % miou, logger=logger)
    for i, name in enumerate(S3DIS_CLASS_NAMES):
        print_log('  %10s: %.2f%%' % (name, per_class_iou[i]), logger=logger)
    print_log('============================================================', logger=logger)

    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Loss', avg_loss, epoch)
        val_writer.add_scalar('Metric/mIoU', miou, epoch)

    return metric


def test_s3dis(args, config, logger=None):
    """S3DIS 独立测试"""
    from tools import builder

    if logger is None:
        logger = get_logger(args.log_name)

    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    base_model = nn.DataParallel(base_model).cuda()

    # 统计参数
    for name, param in base_model.module.named_parameters():
        if any(p in name for p in ['gft_adapter', 'propagation_', 'dgcnn_pro', 'conv1', 'bn1', 'drop1', 'conv2', 'eigen_gate']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base_model.parameters())
    print_log('>> Trainable: %.2fM / %.2fM (%.2f%%)' % (trainable/1e6, total/1e6, trainable/total*100), logger=logger)

    metric = IoU_Metric(config.model.NUM_CLASSES)
    npoints = config.npoints
    num_classes = config.model.NUM_CLASSES

    base_model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            points = data['points'].cuda()
            seg = data['seg'].cuda()

            if points.size(1) > npoints:
                fps_idx = pointnet2_utils.furthest_point_sample(points[:, :, :3], npoints)
                points = pointnet2_utils.gather_operation(
                    points.transpose(1, 2).contiguous(), fps_idx
                ).transpose(1, 2).contiguous()
                fps_idx_expand = fps_idx.expand(-1, -1, seg.shape[-1])
                seg = torch.gather(seg.unsqueeze(1), 1, fps_idx_expand).squeeze(1)

            ret = base_model(points)
            pred = ret.argmax(-1)
            metric.update(pred, seg)

            if idx % 20 == 0:
                print_log('[Test] BATCH: %d/%d' % (idx, len(test_dataloader)), logger=logger)

    miou, per_class_iou = metric.compute()

    print_log('============================================================', logger=logger)
    print_log('[Test] mIoU: %.2f%%' % miou, logger=logger)
    for i, name in enumerate(S3DIS_CLASS_NAMES):
        print_log('  %10s: %.2f%%' % (name, per_class_iou[i]), logger=logger)
    print_log('============================================================', logger=logger)
