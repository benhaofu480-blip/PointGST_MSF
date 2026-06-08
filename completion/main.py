from tools import run_net
from tools import test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter


def main():
    import sys
    print('[train] main() started (file log is created right after args parse)', file=sys.stderr, flush=True)

    args = parser.get_args()

    # torchrun 通过环境变量注入 LOCAL_RANK，需覆盖 argparse 默认值
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True

    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)

        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    print(f'[train] file log: {os.path.abspath(log_file)}', file=sys.stderr, flush=True)

    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None

    config = get_config(args, logger=logger)

    if hasattr(config, 'pretrained_ckpt') and config.pretrained_ckpt:
        if args.start_ckpts is None:
            args.start_ckpts = config.pretrained_ckpt
            logger.info(f'Loading pretrained checkpoint from config: {args.start_ckpts}')
        else:
            logger.info(f'Using command-line start_ckpts: {args.start_ckpts} (ignoring config)')

    if args.model == 'pgst':
        config.model.NAME = 'AdaPoinTr_PGST'
    elif args.model == 'pcsa':
        config.model.NAME = 'AdaPoinTr_PGST'
        config.model.encoder_config.adapter_mode = 'pcsa'
        if hasattr(config.model.encoder_config, 'use_msf'):
            delattr(config.model.encoder_config, 'use_msf')
    elif args.model == 'msf':
        config.model.NAME = 'AdaPoinTr_PGST'
        config.model.encoder_config.use_msf = True
        config.model.encoder_config.adapter_mode = 'msf'
        if hasattr(config.model, 'loss_config'):
            delattr(config.model, 'loss_config')
    elif args.model == 'decouple':
        config.model.NAME = 'AdaPoinTr_PGST'
        config.model.encoder_config.use_msf = False
        config.model.encoder_config.adapter_mode = 'decouple'
        if hasattr(config.model, 'loss_config'):
            delattr(config.model, 'loss_config')
    elif args.model == 'linear':
        config.model.NAME = 'AdaPoinTr'
    elif args.model == 'fft':
        config.model.NAME = 'AdaPoinTr'
    else:
        config.model.NAME = 'AdaPoinTr'
        config.optimizer.part = 'all'

    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs

    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)

    logger.info(f'Distributed training: {args.distributed}')

    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic)

    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()

    if args.test:
        test_net(args, config)
    else:
        run_net(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()
