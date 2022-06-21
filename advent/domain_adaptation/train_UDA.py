# --------------------------------------------------------
# Domain adpatation training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import os
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.utils import data

from advent.model.discriminator import get_fc_discriminator, get_classifier, PixelDiscriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss, loss_const
from advent.utils.loss import entropy_loss, IW_MaxSquareloss, soft_label_cross_entropy
from advent.utils.func import prob_2_entropy
from advent.utils.viz_segmask import colorize_mask
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cs_labels = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'light', 'sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motocycle', 'bicycle']


def train_advent(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True
    
    # DISCRIMINATOR NETWORK
    # feature-level
    d_aux = get_fc_discriminator(num_classes=num_classes)
    try:
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM[:-4] + "D_aux.pth")
    except:
        saved_state_dict = None
    if saved_state_dict:
        try:
            d_aux.load_state_dict(saved_state_dict)
        except:
            new_params = d_aux.state_dict().copy()
            for i in saved_state_dict:
                new_params[i[7:]]= saved_state_dict[i] 
            d_aux.load_state_dict(new_params)
    d_aux.train()
    d_aux.to(device)

    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    try:
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM[:-4] + "D_main.pth")
    except:
        saved_state_dict = None
    if saved_state_dict:
        try:
            d_main.load_state_dict(saved_state_dict)
        except:
            new_params = d_main.state_dict().copy()
            for i in saved_state_dict:
                new_params[i[7:]]= saved_state_dict[i] 
            d_main.load_state_dict(new_params)
    d_main.train()
    d_main.to(device)
    if cfg.distributed:
        d_aux = nn.parallel.DistributedDataParallel(d_aux, device_ids=[cfg.GPU_ID])
        d_main = nn.parallel.DistributedDataParallel(d_main, device_ids=[cfg.GPU_ID])
    # OPTIMIZERS
    # segnet's optimizer
        optimizer = optim.SGD(model.module.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                            lr=cfg.TRAIN.LEARNING_RATE,
                            momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        # discriminators' optimizers
        optimizer_d_aux = optim.Adam(d_aux.module.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                    betas=(0.9, 0.99))
        optimizer_d_main = optim.Adam(d_main.module.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                    betas=(0.9, 0.99))
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                            lr=cfg.TRAIN.LEARNING_RATE,
                            momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        # discriminators' optimizers
        optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                    betas=(0.9, 0.99))
        optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                    betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = iter(trainloader)
    len_train_source = len(trainloader_iter)
    targetloader_iter = iter(targetloader)
    len_train_target = len(targetloader_iter)
    for i_iter in tqdm(range(cfg.TRAIN.INIT_ITER, cfg.TRAIN.EARLY_STOP + 1)):
        if i_iter % len_train_source == 0:
            trainloader_iter = iter(trainloader)
        if i_iter % len_train_target == 0:
            targetloader_iter = iter(targetloader)
        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source
        batch = trainloader_iter.next()
        images_source, labels, _, _ = batch
        pred_src_aux, pred_src_main, _ = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux, _ = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main, class_loss = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # adversarial training to fool the discriminator
        batch = targetloader_iter.next()
        images, _, _, _ = batch
        pred_trg_aux, pred_trg_main, _ = model(images.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = interp_target(pred_trg_aux)
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
        else:
            loss_adv_trg_aux = 0
        pred_trg_main = interp_target(pred_trg_main)
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_adv_trg_main = bce_loss(d_out_main, source_label)
        loss = ( cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)
        #loss = loss
        loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
            loss_d_aux = bce_loss(d_out_aux, source_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_d_aux = bce_loss(d_out_aux, target_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main}
        print_losses(current_losses, i_iter)
        per_class_loss_main ={'loss_'+cs_labels[i]+'_main': class_loss[i] for i in range(19)} 
        current_losses.update(per_class_loss_main)
        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0 and cfg.rank==0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard and cfg.rank==0:
            log_losses_tensorboard(writer, current_losses, i_iter)
            log_losses_conjunction_tensorboard(writer, per_class_loss_main, i_iter)
            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')

def train_adapt_segnet(model, trainloader, targetloader, cfg):
    ''' UDA training with adapt_segnet
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True
    
    # DISCRIMINATOR NETWORK
    # feature-level
    d_aux = get_fc_discriminator(num_classes=num_classes)
    try:
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM[:-4] + "D_aux.pth")
    except:
        saved_state_dict = None
    if saved_state_dict:
        try:
            d_aux.load_state_dict(saved_state_dict)
        except:
            new_params = d_aux.state_dict().copy()
            for i in saved_state_dict:
                new_params[i[7:]]= saved_state_dict[i] 
            d_aux.load_state_dict(new_params)
    d_aux.train()
    d_aux.to(device)

    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    try:
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM[:-4] + "D_main.pth")
    except:
        saved_state_dict = None
    if saved_state_dict:
        try:
            d_main.load_state_dict(saved_state_dict)
        except:
            new_params = d_main.state_dict().copy()
            for i in saved_state_dict:
                new_params[i[7:]]= saved_state_dict[i] 
            d_main.load_state_dict(new_params)
    d_main.train()
    d_main.to(device)
    if cfg.distributed:
        d_aux = nn.parallel.DistributedDataParallel(d_aux, device_ids=[cfg.GPU_ID])
        d_main = nn.parallel.DistributedDataParallel(d_main, device_ids=[cfg.GPU_ID])
    # OPTIMIZERS
    # segnet's optimizer
        optimizer = optim.SGD(model.module.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                            lr=cfg.TRAIN.LEARNING_RATE,
                            momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        # discriminators' optimizers
        optimizer_d_aux = optim.Adam(d_aux.module.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                    betas=(0.9, 0.99))
        optimizer_d_main = optim.Adam(d_main.module.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                    betas=(0.9, 0.99))
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                            lr=cfg.TRAIN.LEARNING_RATE,
                            momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        # discriminators' optimizers
        optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                    betas=(0.9, 0.99))
        optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                    betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = iter(trainloader)
    len_train_source = len(trainloader_iter)
    targetloader_iter = iter(targetloader)
    len_train_target = len(targetloader_iter)
    for i_iter in tqdm(range(cfg.TRAIN.INIT_ITER, cfg.TRAIN.EARLY_STOP + 1)):
        if i_iter % len_train_source == 0:
            trainloader_iter = iter(trainloader)
        if i_iter % len_train_target == 0:
            targetloader_iter = iter(targetloader)
        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        if cfg.distributed:
            for param in d_aux.module.parameters():
                param.requires_grad = False
            for param in d_main.module.parameters():
                param.requires_grad = False
        else:
            for param in d_aux.parameters():
                param.requires_grad = False
            for param in d_main.parameters():
                param.requires_grad = False
        # train on source
        batch = trainloader_iter.next()
        images_source, labels, _, _ = batch
        pred_src_aux, pred_src_main, _ = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux, _ = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main, class_loss = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # adversarial training to fool the discriminator
        batch = targetloader_iter.next()
        images, _, _, _ = batch
        pred_trg_aux, pred_trg_main, _ = model(images.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = interp_target(pred_trg_aux)
            d_out_aux = d_aux(F.softmax(pred_trg_aux))
            loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
        else:
            loss_adv_trg_aux = 0
        pred_trg_main = interp_target(pred_trg_main)
        d_out_main = d_main(F.softmax(pred_trg_main))
        loss_adv_trg_main = bce_loss(d_out_main, source_label)
        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)
        #loss = loss
        loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        if cfg.distributed:
            for param in d_aux.module.parameters():
                param.requires_grad = True
            for param in d_main.module.parameters():
                param.requires_grad = True
        else:
            for param in d_aux.parameters():
                param.requires_grad = True
            for param in d_main.parameters():
                param.requires_grad = True
        # train with source
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()
            d_out_aux = d_aux(F.softmax(pred_src_aux))
            loss_d_aux = bce_loss(d_out_aux, source_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(F.softmax(pred_src_main))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux = d_aux(F.softmax(pred_trg_aux))
            loss_d_aux = bce_loss(d_out_aux, target_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(F.softmax(pred_trg_main))
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main}
        print_losses(current_losses, i_iter)
        per_class_loss_main ={'loss_'+cs_labels[i]+'_main': class_loss[i] for i in range(19)} 
        current_losses.update(per_class_loss_main)
        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0 and cfg.rank==0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard and cfg.rank==0:
            log_losses_tensorboard(writer, current_losses, i_iter)
            log_losses_conjunction_tensorboard(writer, per_class_loss_main, i_iter)
            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')


def train_maxsquare(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    #Alignment configuration 
    loss_max_square = IW_MaxSquareloss()
    threshold = 0.95
    target_hard_loss = nn.CrossEntropyLoss(ignore_index= 255)
    # OPTIMIZERS
    # segnet's optimizer
    if cfg.distributed:
        optimizer = optim.SGD(model.module.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                            lr=cfg.TRAIN.LEARNING_RATE,
                            momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                            lr=cfg.TRAIN.LEARNING_RATE,
                            momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)




    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    trainloader_iter = iter(trainloader)
    len_train_source = len(trainloader_iter)
    targetloader_iter = iter(targetloader)
    len_train_target = len(targetloader_iter)
    for i_iter in tqdm(range(cfg.TRAIN.INIT_ITER, cfg.TRAIN.EARLY_STOP + 1)):
        if i_iter % len_train_source == 0:
            trainloader_iter = iter(trainloader)
        if i_iter % len_train_target == 0:
            targetloader_iter = iter(targetloader)

        # reset optimizers
        optimizer.zero_grad()
    
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)

        # UDA Training
        # train on source
        batch = trainloader_iter.next()
        images_source, labels, _, _ = batch
        pred_src_aux, pred_src_main, _ = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux, _ = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main, class_loss = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # Train with target
        batch = targetloader_iter.next()
        images, _, _, _ = batch
        pred_trg_aux, pred_trg_main, _ = model(images.cuda(device))
        pred_P_2 = None
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = interp_target(pred_trg_aux)
            pred_P_2 = F.softmax(pred_trg_aux, dim=1)

        pred_trg_main = interp_target(pred_trg_main)
        pred_P  = F.softmax(pred_trg_main, dim=1)

        label = pred_P
        if cfg.TRAIN.MULTI_LEVEL: 
            label_2 = pred_P_2

        maxpred, argpred = torch.max(pred_P.detach(), dim=1)
        if cfg.TRAIN.MULTI_LEVEL: 
            maxpred_2, argpred_2 = torch.max(pred_P_2.detach(), dim=1)
        
        loss_target = loss_max_square(pred_trg_main, label)# label_tgt.cuda(device))

        if cfg.TRAIN.MULTI_LEVEL:
            pred_c = (pred_P+pred_P_2)/2
            maxpred_c, argpred_c = torch.max(pred_c, dim=1)
            mask = (maxpred > threshold) | (maxpred_2 > threshold)

            label_2 = torch.where(mask, argpred_c, torch.ones(1).to(device, dtype=torch.long)*255)
            loss_target_aux = target_hard_loss(pred_trg_aux, label_2)# , label_tgt.cuda(device))
        else:
            loss_target_aux = 0

        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_target
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_target_aux)
        loss = loss
        loss.backward()

        optimizer.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_target': loss_target,
                          'loss_target_aux': loss_target_aux}
        print_losses(current_losses, i_iter)
        per_class_loss_main ={'loss_'+cs_labels[i]+'_main': class_loss[i] for i in range(19)} 
        current_losses.update(per_class_loss_main)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0 and cfg.rank==0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard and cfg.rank==0:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')

def train_FADA(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    
    model.train()
    cudnn.benchmark = True
    cudnn.enabled = True

    #Alignment configuration 
    num_features = 2048
    mid_nc=256
    model_D = PixelDiscriminator(num_features, mid_nc, num_classes=cfg.NUM_CLASSES)
    model_D.to(cfg.GPU_ID)
    try:
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM[:-4] + "D.pth")
    except:
        saved_state_dict = None
    if saved_state_dict:
        try:
            model_D.load_state_dict(saved_state_dict)
        except:
            new_params = model_D.state_dict().copy()
            for i in saved_state_dict:
                new_params[i[7:]]= saved_state_dict[i] 
            model_D.load_state_dict(new_params)

    model_D_aux = PixelDiscriminator(1024, mid_nc, num_classes=cfg.NUM_CLASSES)
    model_D_aux.to(cfg.GPU_ID)
    model_D.train()
    model_D_aux.train()
    target_hard_loss = nn.CrossEntropyLoss(ignore_index=255)
    # OPTIMIZERS
    # segnet's optimizer
    if cfg.distributed:
        model_D = nn.parallel.DistributedDataParallel(model_D, device_ids=[cfg.GPU_ID])
        model_D_aux = nn.parallel.DistributedDataParallel(model_D_aux, device_ids=[cfg.GPU_ID])
        optimizer = optim.SGD(model.module.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                            lr=cfg.TRAIN.LEARNING_RATE,
                            momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        optimizer_d_main = optim.Adam(model_D.module.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))
        optimizer_d_aux = optim.Adam(model_D_aux.module.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))
        torch.autograd.set_detect_anomaly(True)
        
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                            lr=cfg.TRAIN.LEARNING_RATE,
                            momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        optimizer_d_main = optim.Adam(model_D.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))
        optimizer_d_aux = optim.Adam(model_D_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))
    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)
    # labels for adversarial training
    trainloader_iter = iter(trainloader)
    source_len = len(trainloader_iter)
    targetloader_iter = iter(targetloader)
    target_len = len(targetloader_iter)
    for i_iter in tqdm(range(cfg.TRAIN.INIT_ITER, cfg.TRAIN.EARLY_STOP + 1)):

        if i_iter % source_len == 0 and i_iter!= 0:
            trainloader_iter = iter(trainloader)
        if i_iter % target_len == 0 and i_iter!= 0:
            targetloader_iter = iter(targetloader)

        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_main.zero_grad()
        optimizer_d_aux.zero_grad()
        
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        batch = trainloader_iter.next()
        images_source, labels, _, _ = batch
        src_size = images_source.shape[-2:]

        batch = targetloader_iter.next()
        images, _, _, _ = batch
        tgt_size = images.shape[-2:]

        # train on source
        pred_src_aux, pred_src_main, features = model(images_source.cuda(non_blocking=True))
        
        temperature = 1.8
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            pred_src_aux = pred_src_aux.div(temperature)
            loss_seg_src_aux, _ = loss_calc(pred_src_aux, labels, device)
            src_soft_label_aux = F.softmax(pred_src_aux, dim=1).detach()
            src_soft_label_aux[src_soft_label_aux>0.9] = 0.9
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        
        pred_src_main = pred_src_main.div(temperature)
        loss_seg_src_main, class_loss = loss_calc(pred_src_main, labels, device)

        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()
        

        # generate soft labels
        src_soft_label = F.softmax(pred_src_main, dim=1).detach()
        src_soft_label[src_soft_label>0.9] = 0.9
        
        # Train with target
        tgt_pred_aux, tgt_pred, tgt_fea = model(images.cuda(non_blocking=True))
        pred_tgt_main_full = interp_target(tgt_pred)
        pred_tgt_main = pred_tgt_main_full.div(temperature)
        tgt_soft_label = F.softmax(pred_tgt_main, dim=1)
        loss_target_entp_main = entropy_loss(tgt_soft_label)
        
        tgt_soft_label_detached = tgt_soft_label.clone().detach()
        tgt_soft_label_detached[tgt_soft_label_detached>0.9] = 0.9
        

        tgt_D_pred = model_D(tgt_fea[0], tgt_size)
        loss_adv_tgt = (soft_label_cross_entropy(tgt_D_pred, torch.cat((tgt_soft_label_detached, torch.zeros_like(tgt_soft_label_detached)), dim=1)))
        if cfg.TRAIN.MULTI_LEVEL:
            tgt_D_pred_aux = model_D_aux(tgt_fea[1], tgt_size)
            pred_tgt_aux = interp_target(tgt_pred_aux)
            pred_tgt_aux = pred_tgt_aux.div(temperature)
            
            tgt_soft_label_aux = F.softmax(pred_tgt_aux, dim=1)
            loss_target_entp_aux = entropy_loss(tgt_soft_label_aux)
            tgt_soft_label_aux_detached = tgt_soft_label_aux.clone().detach()
            tgt_soft_label_aux_detached[tgt_soft_label_aux_detached>0.9] = 0.9

            loss_aux_tgt = soft_label_cross_entropy(tgt_D_pred_aux, torch.cat((tgt_soft_label_aux_detached, torch.zeros_like(tgt_soft_label_aux_detached)), dim=1))
        else:
            loss_aux_tgt = 0
            loss_target_entp_aux = 0
    
        loss_adv = cfg.TRAIN.LAMBDA_ADV_MAIN*(loss_adv_tgt + loss_target_entp_main) + cfg.TRAIN.LAMBDA_ADV_AUX*(loss_aux_tgt + loss_target_entp_aux)
        loss_adv.backward()
        optimizer.step()
        
        optimizer_d_main.zero_grad()
        optimizer_d_aux.zero_grad()
        
        
        src_D_pred = model_D(features[0].detach(), src_size)
        loss_D_src = 0.5*soft_label_cross_entropy(src_D_pred, torch.cat((src_soft_label, torch.zeros_like(src_soft_label)), dim=1))
        loss_D_src.backward()

        tgt_D_pred = model_D(tgt_fea[0].detach(), tgt_size)
        loss_D_tgt = 0.5*soft_label_cross_entropy(tgt_D_pred, torch.cat((torch.zeros_like(tgt_soft_label_detached), tgt_soft_label_detached), dim=1))
        loss_D_tgt.backward()
        if cfg.TRAIN.MULTI_LEVEL:
            src_D_pred_aux = model_D_aux(features[1].detach(), src_size)
            loss_D_src_aux = 0.5*soft_label_cross_entropy(src_D_pred_aux, torch.cat((src_soft_label_aux, torch.zeros_like(src_soft_label_aux)), dim=1))
            loss_D_src_aux.backward()

            tgt_D_pred_aux = model_D_aux(tgt_fea[1].detach(), tgt_size)
            loss_D_tgt_aux = 0.5*soft_label_cross_entropy(tgt_D_pred_aux, torch.cat((torch.zeros_like(tgt_soft_label_aux_detached), tgt_soft_label_aux_detached), dim=1))
            loss_D_tgt_aux.backward()
        else:
            loss_D_src_aux = 0
            loss_D_tgt_aux = 0
        
        optimizer_d_main.step()
        optimizer_d_aux.step()
        
        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_tgt': loss_adv_tgt,
                          'loss_adv_tgt_aux': loss_aux_tgt,
                          'loss_D_src': loss_D_src,
                          'loss_D_tgt': loss_D_tgt,
                          'loss_D_src_aux': loss_D_src_aux,
                          'loss_D_tgt_aux': loss_D_tgt_aux,
                          'loss_target_entp_aux': loss_target_entp_aux,
                          'loss_target_entp_main': loss_target_entp_main}
        print_losses(current_losses, i_iter)

        per_class_loss_main ={'loss_'+cs_labels[i]+'_main': class_loss[i] for i in range(19)} 
        
        current_losses.update(per_class_loss_main)
        

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0 and cfg.rank==0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(model_D.state_dict(), snapshot_dir / f'model_{i_iter}_model_D.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()
        
        # Visualize with tensorboard
        if viz_tensorboard and cfg.rank==0:
            log_losses_conjunction_tensorboard(writer, per_class_loss_main, i_iter)
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_tgt_main_full, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')
        

def train_CL(model, source_dataset, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    device = 0# cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)
    # Road, Car, Veg, BD, SK, Per, Moto, Rider,  Sidewalk, Bus, Truck, Terrain, Light, Bycicle, pole, fence, wall ,sign, train 
    if cfg.TRAIN.ORDER == "Marginal":
        sort_order =[0, 13, 8, 2, 10, 11, 17,12, 14, 9, 4, 1, 6,18,3,5,7,15,16,  255] 
    elif cfg.TRAIN.ORDER == "Pre_defined":
        sort_order = [8, 10, 13, 2, 0, 11, 12, 6, 15, 17, 9, 5, 18, 3, 4, 1, 16, 14, 7, 255]
    elif cfg.TRAIN.ORDER == "Borders":
        sort_order = [0, 2, 10, 1, 8, 13, 9, 3, 14, 5, 4, 15, 11, 6, 7, 16, 17, 12, 18, 255]
    elif cfg.TRAIN.ORDER == "Images":
        sort_order = [2, 8, 0, 5, 10, 1, 9, 3, 13, 11, 4, 6, 14, 7, 15, 17, 12, 16, 18, 255]


    if cfg.ANTI_CL:
        sort_order.reverse()
        aux = [10] 
        aux.extend(sort_order) 
        sort_order = aux
    if cfg.TRAIN.CLASSES_INCLUDE >= len(sort_order):
        included = None 
    else:
        included = sort_order[:cfg.TRAIN.CLASSES_INCLUDE]  
    # i=cfg.TRAIN.CLASSES_INCLUDE
    #save_after = (i-3)*cfg.TRAIN.INCLUDE_NEW
    # SEGMNETATION NETWORK
    
    #model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True
    model.train()
    print(included)
    # OPTIMIZERS
    # segnet's optimizer
    if cfg.distributed:
        
        optimizer = optim.SGD(model.module.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                            lr=cfg.TRAIN.LEARNING_RATE,
                            momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                                lr=cfg.TRAIN.LEARNING_RATE,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)


    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    if cfg.TRAIN.PACING == "linear":
        cfg.TRAIN.EARLY_STOP = int(cfg.TRAIN.EARLY_STOP/(19 - len(included)))
        cfg.TRAIN.MAX_ITERS = cfg.TRAIN.EARLY_STOP*2
    str_included = [cs_labels[inclusions] for inclusions in included]
    source_dataset.include_class(str_included)
    
    if cfg.TRAIN.PACING == "proportional":
        n_iters_total = cfg.TRAIN.EARLY_STOP
        cfg.TRAIN.EARLY_STOP = source_dataset.get_n_iters(str_included, n_iters_total) 
        cfg.TRAIN.MAX_ITERS = cfg.TRAIN.EARLY_STOP*2
    i = 1
    prev = 0
    for inclusions in sort_order[cfg.TRAIN.CLASSES_INCLUDE:]:
        if cfg.distributed:
            source_dataset_sampler = torch.utils.data.distributed.DistributedSampler(source_dataset,
                                                                        num_replicas=cfg.world_size,
                                                                        rank=cfg.rank, shuffle=True)
            trainloader = data.DataLoader(source_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=False,
                                    pin_memory=True,
                                    sampler=source_dataset_sampler) 
        else:
            trainloader = data.DataLoader(source_dataset,
                                        batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
                                        num_workers=cfg.NUM_WORKERS,
                                        shuffle=True,
                                        pin_memory=True)
        
        trainloader_iter = iter(trainloader)
        len_train_source = len(trainloader_iter)
        if cfg.TRAIN.PACING == "1view":
            cfg.TRAIN.EARLY_STOP = len_train_source
            cfg.TRAIN.MAX_ITERS = len_train_source*2

        #worst = open("files/loss"+cfg.TRAIN.ORDER+str(included[-1]), "w")
        
        for i_iter in tqdm(range(cfg.TRAIN.INIT_ITER, cfg.TRAIN.EARLY_STOP)):
            if i_iter % len_train_source == 0 and i_iter != 0:
                trainloader_iter = iter(trainloader)
            # reset optimizers
            optimizer.zero_grad()
            # adapt LR if needed
            adjust_learning_rate(optimizer, i_iter, cfg)
        

            # UDA Training
            # only train segnet. Don't accumulate grads in disciminators
            # train on source
            batch = trainloader_iter.next()
            images_source, labels, _, name = batch
            #print(images_source)
            pred_src_aux, pred_src_main, _ = model(images_source.cuda(device))
            if cfg.TRAIN.MULTI_LEVEL:
                pred_src_aux = interp(pred_src_aux)
                loss_seg_src_aux, class_loss_int = loss_calc(pred_src_aux, labels, device,included=included)
            else:
                loss_seg_src_aux = 0
            pred_src_main = interp(pred_src_main)
            loss_seg_src_main, class_loss = loss_calc(pred_src_main, labels, device,included=included)
            loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                    + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
            loss.backward()
            if loss < 0:
                worst.write(str(name) + "\t" + str(loss.detach().cpu().numpy()))
                worst.write('\n')
            optimizer.step()

            current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                            'loss_seg_src_main': loss_seg_src_main
                            }
            
            #per_class_loss_int ={'loss_'+cs_labels[i]+'_aux': class_loss_int[i] for i in range(19)} 
            per_class_loss_main ={'loss_'+cs_labels[i]+'_main': class_loss[i] for i in range(19)} 
            current_losses.update(per_class_loss_main)
            #print_losses(current_losses, i_iter)
            
        
        
            sys.stdout.flush()

            # Visualize with tensorboard
            if viz_tensorboard and cfg.rank==0:
                log_losses_tensorboard(writer, current_losses, i_iter+ prev)
                log_losses_conjunction_tensorboard(writer, per_class_loss_main, i_iter + prev)
                if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                    draw_in_tensorboard(writer, images_source, i_iter +prev, pred_src_main, num_classes, 'S')
        prev += i_iter
        print('taking snapshot ...')
        print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
        snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
        if cfg.rank == 0:
            try:
                torch.save(model.module.state_dict(), snapshot_dir / f'model_{i}.pth')
            except:
                torch.save(model.state_dict(), snapshot_dir / f'model_{i}.pth')
        # worst.close()
        source_dataset.include_class(cs_labels[inclusions])
        if cfg.TRAIN.PACING == "proportional":
            cfg.TRAIN.EARLY_STOP = source_dataset.get_n_iters(cs_labels[inclusions], n_iters_total) 
            cfg.TRAIN.MAX_ITERS = cfg.TRAIN.EARLY_STOP*2
        included.append(inclusions)
        i += 1


def train_source_only(model, trainloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)
    # Road, Car, Veg, BD, SK, Per, Moto, Rider,  Sidewalk, Bus, Truck, Terrain, Light, Bycicle, pole, fence, wall ,sign, train 
    included = None
    # i=cfg.TRAIN.CLASSES_INCLUDE
    #save_after = (i-3)*cfg.TRAIN.INCLUDE_NEW
    # SEGMNETATION NETWORK
    model.train()
    #model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True
    
    print(included)
    # OPTIMIZERS
    # segnet's optimizer
    if cfg.distributed:
        optimizer = optim.SGD(model.module.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                            lr=cfg.TRAIN.LEARNING_RATE,
                            momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                                lr=cfg.TRAIN.LEARNING_RATE,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)


    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)

   
    trainloader_iter = iter(trainloader)
    len_train_source = len(trainloader_iter)
    worst = open("files/loss_complete", "w")
    
    for i_iter in tqdm(range(cfg.TRAIN.INIT_ITER, cfg.TRAIN.EARLY_STOP + 1)):
        if i_iter % len_train_source == 0:
            trainloader_iter = iter(trainloader)
        # reset optimizers
        optimizer.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
    

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        # train on source
        batch = trainloader_iter.next()
        images_source, labels, _, name = batch
        #print(images_source)
        pred_src_aux, pred_src_main, _ = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux, class_loss_int = loss_calc(pred_src_aux, labels, device,included=included)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main, class_loss = loss_calc(pred_src_main, labels, device,included=included)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()
        if loss > 0:
            worst.write(str(name) + "\t" + str(loss.detach().cpu().numpy()))
            worst.write('\n')
        optimizer.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                        'loss_seg_src_main': loss_seg_src_main
                        }
        
        #per_class_loss_int ={'loss_'+cs_labels[i]+'_aux': class_loss_int[i] for i in range(19)} 
        per_class_loss_main ={'loss_'+cs_labels[i]+'_main': class_loss[i] for i in range(19)} 
        current_losses.update(per_class_loss_main)
        #print_losses(current_losses, i_iter)
        
    
    
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard and cfg.rank==0:
            log_losses_tensorboard(writer, current_losses, i_iter)
            log_losses_conjunction_tensorboard(writer, per_class_loss_main, i_iter )
            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')
    print('taking snapshot ...')
    print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
    snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
    try:
        torch.save(model.module.state_dict(), snapshot_dir / f'model_1.pth')
    except:
        torch.save(model.state_dict(), snapshot_dir / f'model_1.pth')
    worst.close()


def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    if pred_main is not None:
        grid_image = make_grid(images[0].clone().cpu().data, 3, normalize=True)
        writer.add_image(f'Image - {type_}', grid_image, i_iter)
        grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
            np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                    axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                            normalize=False, range=(0, 255))
        writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

        output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
        output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                            keepdims=False)
        grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                            range=(0, np.log2(num_classes)))
        writer.add_image(f'Entropy - {type_}', grid_image, i_iter)
    else:
        grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(images.numpy()[0]).convert('RGB')).transpose(2, 0, 1)), 3,
                            normalize=False, range=(0, 255))
        writer.add_image(f'Pseudolabel - {type_}', grid_image, i_iter)
    


def train_minent(model, trainloader, targetloader, cfg):
    ''' UDA training with minEnt
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    # segnet's optimizer
    if cfg.distributed:
        optimizer = optim.SGD(model.module.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                            lr=cfg.TRAIN.LEARNING_RATE,
                            momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                            lr=cfg.TRAIN.LEARNING_RATE,
                            momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    trainloader_iter = iter(trainloader)
    len_train_source = len(trainloader_iter)
    targetloader_iter = iter(targetloader)
    len_train_target = len(targetloader_iter)
    for i_iter in tqdm(range(cfg.TRAIN.INIT_ITER, cfg.TRAIN.EARLY_STOP + 1)):
        if i_iter % len_train_source == 0:
            trainloader_iter = iter(trainloader)
        if i_iter % len_train_target == 0:
            targetloader_iter = iter(targetloader)
        # reset optimizers
        optimizer.zero_grad()

        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)

        # UDA Training
        # train on source
        batch = trainloader_iter.next()
        images_source, labels, _, _ = batch
        pred_src_aux, pred_src_main, _ = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux, _ = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main, class_loss = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # adversarial training with minent
        batch = targetloader_iter.next()
        images, _, _, _ = batch
        pred_trg_aux, pred_trg_main, _ = model(images.cuda(device))
        pred_trg_aux = interp_target(pred_trg_aux)
        pred_trg_main = interp_target(pred_trg_main)
        pred_prob_trg_aux = F.softmax(pred_trg_aux)
        pred_prob_trg_main = F.softmax(pred_trg_main)

        loss_target_entp_aux = entropy_loss(pred_prob_trg_aux)
        loss_target_entp_main = entropy_loss(pred_prob_trg_main)
        loss = (cfg.TRAIN.LAMBDA_ENT_AUX * loss_target_entp_aux
                + cfg.TRAIN.LAMBDA_ENT_MAIN * loss_target_entp_main)
        loss.backward()
        optimizer.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_ent_aux': loss_target_entp_aux,
                          'loss_ent_main': loss_target_entp_main}

        print_losses(current_losses, i_iter)
        per_class_loss_main ={'loss_'+cs_labels[i]+'_main': class_loss[i] for i in range(19)} 
        current_losses.update(per_class_loss_main)
        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(),
                       osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')



def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings) 
    tqdm.write(f'iter = {i_iter} {full_string}')


def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)
def log_losses_conjunction_tensorboard(writer, current_losses, i_iter):
    writer.add_scalars('per_classes', current_losses, i_iter)

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def train_domain_adaptation(model, trainloader, targetloader, cfg, source_dataset):
    if cfg.TRAIN.DA_METHOD == 'MinEnt':
        train_minent(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'AdvEnt':
        train_advent(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'CL':
        trainloader = None
        targetloader = None
        train_CL(model, source_dataset, cfg)   
    elif cfg.TRAIN.DA_METHOD == "adapt_segnet":
        train_adapt_segnet(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == "maxsquare":
        train_maxsquare(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == "FADA":
        train_FADA(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == "Source_only":
        train_source_only(model, trainloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")
