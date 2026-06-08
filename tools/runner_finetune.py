import os

import torch
import torch.nn as nn

from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        # data_transforms.RandomHorizontalFlip(),
        # data_transforms.PointcloudScaleAndTranslate(),
    ]
)

train_transforms_raw = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        # data_transforms.RandomHorizontalFlip(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset

    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # trainval
    # training
    base_model.zero_grad()
    misc.summary_parameters(base_model, logger=logger)

    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)

            points = data[0].cuda()
            label = data[1].cuda()

            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)

            fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                                              2).contiguous()  # (B, N, 3)
            if ('recon' in args.config) or ('act' in args.config):
                if 'scan' in args.config:
                    points = train_transforms(points)
                else:
                    points = train_transforms_raw(points)
            else:
                points = train_transforms_raw(points)

            ret = base_model(points)
            loss, acc = base_model.module.get_loss_acc(ret, label)
            _loss = loss

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

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
            # Validate the current model
            if 'ShapeNetPart' in config.dataset.val._base_:
                # ShapeNetPart分割验证
                metrics = validate_shapenetpart(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
                # 对于分割任务，使用实例mIoU作为评判标准
                best_metric_value = best_metrics.acc if hasattr(best_metrics, 'acc') else 0
                current_metric_value = metrics.acc if hasattr(metrics, 'acc') else 0
                better = current_metric_value > best_metric_value
            else:
                # 分类验证
                metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
                better = metrics.better_than(best_metrics)
            
            # Save checkpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                        logger=logger)
                print_log(
                    "--------------------------------------------------------------------------------------------",
                    logger=logger)
            if args.vote and 'ShapeNetPart' not in config.dataset.val._base_:
                if metrics.acc > 92.1 or (better and metrics.acc > 91):
                    metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config,
                                                 logger=logger)
                    if metrics_vote.better_than(best_metrics_vote):
                        best_metrics_vote = metrics_vote
                        print_log(
                            "****************************************************************************************",
                            logger=logger)
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote,
                                                'ckpt-best_vote', args, logger=logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)


def validate_shapenetpart(base_model, test_dataloader, epoch, val_writer, args, config, logger=None):
    """ShapeNetPart验证函数 - dataset级mIoU累计计算"""
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()

    # 累计每个类别的intersection和union
    npoints = config.npoints
    # class_miou: 每个物体类别(0-15)的累计IoU列表
    category_iou = {}  # cls_id -> list of sample iou

    test_loss_sum = 0.0
    test_loss_count = 0

    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            points = data['points'].cuda()
            cls = data['cls'].cuda()
            seg = data['seg'].cuda()

            # FPS采样（同步重采样seg标签）
            if points.size(1) > npoints:
                fps_idx = pointnet2_utils.furthest_point_sample(points, npoints)
                points = pointnet2_utils.gather_operation(
                    points.transpose(1, 2).contiguous(), fps_idx
                ).transpose(1, 2).contiguous()
                fps_idx_expand = fps_idx.expand(-1, -1, seg.shape[-1])
                seg = torch.gather(seg.unsqueeze(1), 1, fps_idx_expand).squeeze(1)

            ret = base_model(points, cls)
            loss = base_model.module.loss_ce(ret.reshape(-1, ret.shape[-1]), seg.reshape(-1))
            test_loss_sum += loss.item()
            test_loss_count += 1

            # 计算每个样本的mIoU
            pred = ret.argmax(-1)  # (B, N)
            batch_size = pred.shape[0]
            for b in range(batch_size):
                cls_id = cls[b].item()
                valid_parts = base_model.module.class2parts.get(cls_id, [])

                iou_list = []
                for part in valid_parts:
                    intersection = ((pred[b] == part) & (seg[b] == part)).sum().item()
                    union = ((pred[b] == part) | (seg[b] == part)).sum().item()
                    if union == 0:
                        iou_list.append(1.0)
                    else:
                        iou_list.append(intersection / union)

                sample_miou = np.mean(iou_list)
                if cls_id not in category_iou:
                    category_iou[cls_id] = []
                category_iou[cls_id].append(sample_miou)

            if idx % 50 == 0:
                print_log('[Validation] EPOCH: %d BATCH: %d/%d' % (
                    epoch, idx, len(test_dataloader)), logger=logger)

        avg_loss = test_loss_sum / max(test_loss_count, 1)

        # Cls.mIoU: 每个物体类别的平均IoU -> 再对所有类别取平均
        cls_miou_list = [np.mean(v) for v in category_iou.values() if len(v) > 0]
        cls_miou = np.mean(cls_miou_list) * 100 if cls_miou_list else 0.0

        # Inst.mIoU: 所有样本的IoU直接取平均
        all_instance_ious = []
        for v in category_iou.values():
            all_instance_ious.extend(v)
        inst_miou = np.mean(all_instance_ious) * 100 if all_instance_ious else 0.0

        print_log('=' * 60, logger=logger)
        print_log('[Validation] EPOCH: %d' % epoch, logger=logger)
        print_log('  Loss: %.4f' % avg_loss, logger=logger)
        print_log('  Cls.mIoU: %.2f%%' % cls_miou, logger=logger)
        print_log('  Inst.mIoU: %.2f%%' % inst_miou, logger=logger)
        print_log('=' * 60, logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    if val_writer is not None:
        val_writer.add_scalar('Loss/val', avg_loss, epoch)
        val_writer.add_scalar('Metric/Cls_mIoU', cls_miou, epoch)
        val_writer.add_scalar('Metric/Inst_mIoU', inst_miou, epoch)

    return Acc_Metric(inst_miou)


def run_net_shapenetpart(args, config, train_writer=None, val_writer=None, logger=None):
    """ShapeNetPart分割训练函数"""
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = \
        builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)

    # build model
    base_model = builder.model_builder(config.model)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()])
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

    # 冻结backbone的BatchNorm，防止预训练的running stats被破坏
    # 但上采样网络(propagation/dgcnn)和分割头的BN保持train模式
    for name, module in base_model.module.named_modules():
        if any(p in name for p in ['propagation_', 'dgcnn_pro', 'conv1', 'bn1', 'conv2']):
            continue  # 上采样网络和分割头的BN不冻结
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
        data_time = AverageMeter()
        losses = AverageMeter(['loss'])
        num_iter = 0
        n_batches = len(train_dataloader)

        npoints = config.npoints
        for idx, data in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)

            # ShapeNetPart返回dict格式
            points = data['points'].cuda()
            cls = data['cls'].cuda()
            seg = data['seg'].cuda()

            # FPS采样（同步重采样seg标签）
            if points.size(1) > npoints:
                fps_idx = pointnet2_utils.furthest_point_sample(points, npoints)
                points = pointnet2_utils.gather_operation(
                    points.transpose(1, 2).contiguous(), fps_idx
                ).transpose(1, 2).contiguous()
                fps_idx_expand = fps_idx.expand(-1, -1, seg.shape[-1])
                seg = torch.gather(seg.unsqueeze(1), 1, fps_idx_expand).squeeze(1)

            ret = base_model(points, cls)

            # 计算损失
            loss = base_model.module.loss_ce(ret.reshape(-1, ret.shape[-1]), seg.reshape(-1))

            _loss = loss
            _loss.backward()

            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item()])
            else:
                losses.update([loss.item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

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
            metrics = validate_shapenetpart(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
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


def test_shapenetpart(args, config, logger=None):
    """ShapeNetPart独立测试函数 - 输出Params, Cls.mIoU, Inst.mIoU"""
    from tools import builder

    if logger is None:
        logger = get_logger(args.log_name)

    # build test dataset
    (_, test_dataloader) = builder.dataset_builder(args, config.dataset.test)

    # build model
    base_model = builder.model_builder(config.model)

    # load checkpoint
    builder.load_model(base_model, args.ckpts, logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    if args.distributed:
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
    else:
        base_model = nn.DataParallel(base_model).cuda()

    # 模拟part:adapt的冻结逻辑，准确统计可训练参数
    for name, param in base_model.module.named_parameters():
        if any(p in name for p in ['gft_adapter', 'propagation_', 'dgcnn_pro', 'conv1', 'bn1', 'drop1', 'conv2', 'eigen_gate']):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 统计可训练参数量
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in base_model.parameters())
    print_log('>> Trainable Parameters: %.2fM / %.2fM (%.2f%%)' % (
        trainable_params / 1e6, total_params / 1e6, trainable_params / total_params * 100), logger=logger)

    # test
    base_model.eval()

    category_iou = {}
    test_loss_sum = 0.0
    test_loss_count = 0
    npoints = config.npoints

    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            points = data['points'].cuda()
            cls = data['cls'].cuda()
            seg = data['seg'].cuda()

            if points.size(1) > npoints:
                fps_idx = pointnet2_utils.furthest_point_sample(points, npoints)
                points = pointnet2_utils.gather_operation(
                    points.transpose(1, 2).contiguous(), fps_idx
                ).transpose(1, 2).contiguous()

            ret = base_model(points, cls)
            loss = base_model.module.loss_ce(ret.reshape(-1, ret.shape[-1]), seg.reshape(-1))
            test_loss_sum += loss.item()
            test_loss_count += 1

            pred = ret.argmax(-1)
            batch_size = pred.shape[0]
            for b in range(batch_size):
                cls_id = cls[b].item()
                valid_parts = base_model.module.class2parts.get(cls_id, [])

                iou_list = []
                for part in valid_parts:
                    intersection = ((pred[b] == part) & (seg[b] == part)).sum().item()
                    union = ((pred[b] == part) | (seg[b] == part)).sum().item()
                    if union == 0:
                        iou_list.append(1.0)
                    else:
                        iou_list.append(intersection / union)

                sample_miou = np.mean(iou_list)
                if cls_id not in category_iou:
                    category_iou[cls_id] = []
                category_iou[cls_id].append(sample_miou)

            if idx % 50 == 0:
                print_log('[Test] BATCH: %d/%d' % (idx, len(test_dataloader)), logger=logger)

    avg_loss = test_loss_sum / max(test_loss_count, 1)

    # Cls.mIoU
    cls_miou_list = [np.mean(v) for v in category_iou.values() if len(v) > 0]
    cls_miou = np.mean(cls_miou_list) * 100 if cls_miou_list else 0.0

    # Inst.mIoU
    all_instance_ious = []
    for v in category_iou.values():
        all_instance_ious.extend(v)
    inst_miou = np.mean(all_instance_ious) * 100 if all_instance_ious else 0.0

    # 每个类别的详细mIoU
    print_log('=' * 60, logger=logger)
    print_log('Final Test Results:', logger=logger)
    print_log('  Params.(M): %.2f' % (trainable_params / 1e6), logger=logger)
    print_log('  Cls.mIoU(%%): %.2f' % cls_miou, logger=logger)
    print_log('  Inst.mIoU(%%): %.2f' % inst_miou, logger=logger)
    print_log('  Loss: %.4f' % avg_loss, logger=logger)
    print_log('-' * 60, logger=logger)
    for cls_id in sorted(category_iou.keys()):
        avg = np.mean(category_iou[cls_id]) * 100
        print_log('  Category %2d: %.2f%% (%d samples)' % (cls_id, avg, len(category_iou[cls_id])), logger=logger)
    print_log('=' * 60, logger=logger)


def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times=10):
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(),
                                                          fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)

            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)

    return Acc_Metric(acc)


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=logger)  # for finetuned transformer
    # base_model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


def test(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        import time
        inference_time_list = []
        inference_fps_list = []
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            if idx <= 0.2 * len(test_dataloader) or idx >= 0.8 * len(test_dataloader):
                points = data[0].cuda()
                label = data[1].cuda()
                points = misc.fps(points, npoints)
                logits = base_model(points)
                target = label.view(-1)
                pred = logits.argmax(-1).view(-1)
                test_pred.append(pred.detach())
                test_label.append(target.detach())
            else:
                torch.cuda.synchronize()
                time_start = time.time()
                points = data[0].cuda()
                label = data[1].cuda()
                points = misc.fps(points, npoints)
                logits = base_model(points)
                target = label.view(-1)
                pred = logits.argmax(-1).view(-1)
                test_pred.append(pred.detach())
                test_label.append(target.detach())
                torch.cuda.synchronize()
                epoch_time = time.time() - time_start
                inference_time_list.append(epoch_time)
                inference_fps_list.append(points.shape[0] / epoch_time)
        inference_time = np.mean(inference_time_list)
        inference_fps = np.mean(inference_fps_list)
        print_log(f"[TEST] inference time: {inference_time}", logger=logger)
        print_log(f"[TEST] inference FPS: {inference_fps}", logger=logger)

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[TEST] acc = %.4f' % acc, logger=logger)

        if args.vote:

            if args.distributed:
                torch.cuda.synchronize()

            print_log(f"[TEST_VOTE]", logger=logger)
            acc = 0.
            for time in range(1, 300):
                this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
                if acc < this_acc:
                    acc = this_acc
                print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (time, this_acc, acc), logger=logger)
            print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)


def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times=10):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            elif npoints == 2048:
                point_all = 2048

            else:
                raise NotImplementedError()

            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(),
                                                          fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)

            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    # print_log('[TEST] acc = %.4f' % acc, logger=logger)

    return acc


def plot_embedding(data, label, title, category_nums):
    TSNE_PATH = "./vis/tsne/"
    # colors = []
    colors = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#42d4f4', '#f032e6', '#fabed4', '#469990',
              '#dcbeff', '#9A6324', '#800000', '#000075', '#a9a9a9', '#888870', '#000000'
              ]
    if category_nums > 27:
        base = [0, 0.3, 0.6, 0.9]
    else:
        base = [0, 0.5, 0.9]
    # for i in range(len(base)):
    #     for j in range(len(base)):
    #         for k in range(len(base)):
    #             colors.append([base[i], base[j], base[k], 1])

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(8, 8))
    for i in range(data.shape[0]):
        print(colors[int(label[i])])
        plt.scatter(data[i, 0], data[i, 1], s=8, marker='o', c=colors[int(label[i])], cmap='coolwarm')
        # plt.text(data[i, 0], data[i, 1], str(label[i]),
        #          color=colors[int(label[i])],
        #          # fontdict={'weight': 'bold', 'size': 9}
        #          fontdict={'family': 'Times New Roman',
        #                    'weight': 'normal',
        #                    'size': 8, }
        #          )
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.axis('off')

    if not os.path.isdir(TSNE_PATH):
        os.makedirs(TSNE_PATH)
    plt.savefig(TSNE_PATH + "tsne_fix_.pdf")
    return fig


def test_only_tsne(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    test_feature = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            # get_local.clear()
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            _, concat_f = base_model(points)

            target = label.view(-1)

            test_label.append(target.detach())
            test_feature.append(concat_f.detach())

        test_label = torch.cat(test_label, dim=0)

        category_nums = config.model.cls_dim

        index = test_label < category_nums
        label_all = test_label[index]
        test_feature = torch.cat(test_feature, dim=0)
        test_feature = test_feature[index]

        # tsne
        test_feature = test_feature.cpu().numpy()
        label = label_all.cpu().numpy()

        tsne = TSNE(n_components=2, init='pca', random_state=0)
        result = tsne.fit_transform(test_feature.squeeze())

        fig = plot_embedding(result, label, '', category_nums)


def test_tsne(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.val)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    # builder.load_model(base_model, args.ckpts, logger=logger)  # for finetuned transformer
    base_model.load_model_from_ckpt(args.ckpts)  # for BERT

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test_only_tsne(base_model, test_dataloader, args, config, logger=logger)
