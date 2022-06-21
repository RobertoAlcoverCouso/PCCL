# --------------------------------------------------------
# AdvEnt training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import argparse
import os
import os.path as osp
import pprint
import random
import warnings
import builtins
import numpy as np
import yaml
import torch
from torch.utils import data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
from advent.model.deeplabv2 import get_deeplab_v2
from advent.dataset.gta5 import GTA5DataSet
from advent.dataset.cityscapes import CityscapesDataSet
from advent.domain_adaptation.config import cfg, cfg_from_file
from advent.domain_adaptation.train_UDA import train_domain_adaptation
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz-every-iter", type=int, default=None,
                        help="visualize results.")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    parser.add_argument("-n", "--nodes", default=1, type=int, metavar='N')
    parser.add_argument("-g", "--gpus", default=1, type=int)
    parser.add_argument("-nr", "--nr", default=0, type=int)

    return parser.parse_args()


def main():
    # LOAD ARGS
    args = get_arguments()
    os.environ['MASTER_ADDR'] = '192.168.23.202'
    os.environ['MASTER_PORT'] = '8888'
    print('Called with args:')
    print(args)
    args.world_size = args.nodes * args.gpus
    args.distributed = args.world_size > 1
    if args.distributed:
        if args.nr == 0:
            print('In total: Using {} nodes and {} gpus'.format(args.nodes, args.world_size))
        mp.spawn(main_worker, nprocs=args.gpus, args=(args,))
    else:
        print('Regular training')
        main_worker(gpu=0, args=args)

def main_worker(gpu, args): 
    assert args.cfg is not None, 'Missing cfg file'
    print(gpu)
    cfg_from_file(args.cfg)
    cfg.distributed = args.distributed
    torch.cuda.set_device(gpu)
    cfg.GPU_ID = gpu
    cfg.nr = args.nr
    cfg.world_size = args.world_size
    if args.distributed:
        rank = args.nr * args.gpus + gpu
        print("Starting training in GPU:{} Rank:{} ".format(gpu, rank))
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    else:
        rank = 0
        print("Starting regular training in GPU:{} Rank:{} ".format(gpu, rank))
    cfg.rank = rank
    """
    if rank != 0:
        def print_pass(*args):
            pass

        print('Disabling prints')
        builtins.print = print_pass
        pprint.pprint = print_pass
    """
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'

    if args.exp_suffix:
        cfg.EXP_NAME += f'_{args.exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
    # tensorboard
    if args.tensorboard:
        if cfg.TRAIN.TENSORBOARD_LOGDIR == '':
            cfg.TRAIN.TENSORBOARD_LOGDIR = osp.join(cfg.EXP_ROOT_LOGS, 'tensorboard', cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR, exist_ok=True)
        if args.viz_every_iter is not None:
            cfg.TRAIN.TENSORBOARD_VIZRATE = args.viz_every_iter
    else:
        cfg.TRAIN.TENSORBOARD_LOGDIR = ''
    print('Using config:')
    pprint.pprint(cfg)

    # INIT
    _init_fn = None
    if not args.random_train:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    # LOAD SEGMENTATION NET
    assert osp.exists(cfg.TRAIN.RESTORE_FROM), f'Missing init model {cfg.TRAIN.RESTORE_FROM}'
    if cfg.TRAIN.MODEL == 'DeepLabv2':
        FreezeBN = False # 'FADA' in cfg.EXP_NAME
        model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL, FreezeBN= FreezeBN)
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
        if 'DeepLab_resnet_pretrained_imagenet' in cfg.TRAIN.RESTORE_FROM:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            try:
                model.load_state_dict(saved_state_dict)
            except:
                new_params = model.state_dict().copy()
                for i in saved_state_dict:
                    if FreezeBN and "bn" in i:
                        continue
                    elif i[7:] in new_params:
                        new_params[i[7:]]= saved_state_dict[i] 
                    elif i in new_params:
                        new_params[i] = saved_state_dict[i]  
                model.load_state_dict(new_params)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")
    print('Model loaded')
    #model.cuda()
    # DATALOADERS
    source_dataset = GTA5DataSet(root=cfg.DATA_DIRECTORY_SOURCE,
                                 list_path=cfg.DATA_LIST_SOURCE,
                                 set=cfg.TRAIN.SET_SOURCE,
                                 max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
                                 crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
                                 mean=cfg.TRAIN.IMG_MEAN,
                                 test=False,
                                 files_to_use=cfg.TRAIN.SOURCE_CSV)
    target_dataset = CityscapesDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                       list_path=cfg.DATA_LIST_TARGET,
                                       set=cfg.TRAIN.SET_TARGET,
                                       info_path=cfg.TRAIN.INFO_TARGET,
                                       max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_TARGET,
                                       crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
                                       mean=cfg.TRAIN.IMG_MEAN,
                                       test=False)
    
    #torch.cuda.set_device(gpu)
    model.cuda(cfg.GPU_ID)
    print("distributing")
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.GPU_ID], output_device=cfg.GPU_ID)
        
        source_dataset_sampler = torch.utils.data.distributed.DistributedSampler(source_dataset,
                                                                        num_replicas=args.world_size,
                                                                        rank=rank, shuffle=True)
        target_dataset_sampler = torch.utils.data.distributed.DistributedSampler(target_dataset,
                                                                        num_replicas=args.world_size,
                                                                        rank=rank, shuffle=False)
        source_loader = data.DataLoader(source_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=False,
                                    pin_memory=True,
                                    sampler=source_dataset_sampler) 
        target_loader = data.DataLoader(target_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=False,
                                    pin_memory=True,
                                    sampler=target_dataset_sampler)  
        
    
    else:                                                              
        source_loader = data.DataLoader(source_dataset,
                                        batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
                                        num_workers=cfg.NUM_WORKERS,
                                        shuffle=True,
                                        pin_memory=True,
                                        worker_init_fn=_init_fn)

        
        target_loader = data.DataLoader(target_dataset,
                                        batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
                                        num_workers=cfg.NUM_WORKERS,
                                        shuffle=True,
                                        pin_memory=True,
                                        worker_init_fn=_init_fn)
    """
    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)   
                                 
    """

    
    print("UDA")
    # UDA TRAINING
    train_domain_adaptation(model, source_loader, target_loader, cfg, source_dataset)


if __name__ == '__main__':
    
    main()
