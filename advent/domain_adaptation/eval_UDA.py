# --------------------------------------------------------
# Domain adpatation evaluation
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import os
import os.path as osp
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm
import cv2

from advent.utils import project_root
from advent.utils.func import per_class_iu, fast_hist
from advent.utils.viz_segmask import color_mapping, show
from advent.utils.serialization import pickle_dump, pickle_load
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def evaluate_domain_adaptation( models, test_loader, cfg,
                                fixed_test_size=True,
                                verbose=True, pseudo_loader=False):
    device = cfg.GPU_ID
    interp = None
    if fixed_test_size:
        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)
    # eval
    if cfg.TEST.MODE == 'single':
        eval_single(cfg, models,
                    device, test_loader, interp, fixed_test_size,
                    verbose)
    elif cfg.TEST.MODE == 'best':
        eval_best(cfg, models,
                  device, test_loader, interp, fixed_test_size,
                  verbose)
    elif cfg.TEST.MODE == 'visualize':
        visualize(cfg, models,
                  device, test_loader, interp, fixed_test_size,
                  verbose)
    elif cfg.TEST.MODE == 'pseudolabels':
        generate_pseudolabels(cfg, models,
                  device, pseudo_loader, interp, fixed_test_size,
                  verbose)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")


def eval_single(cfg, models,
                device, test_loader, interp,
                fixed_test_size, verbose):
    assert len(cfg.TEST.RESTORE_FROM) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.TEST.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    for index, batch in tqdm(enumerate(test_loader)):
        image, label, _, name = batch
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            if cfg.TEST.FLIP:
                output_flip = None
            output = None
            for model, model_weight in zip(models, cfg.TEST.MODEL_WEIGHT):
                pred_aux, pred_main, _ = model(image.cuda(device))
                output_ = F.softmax(interp(pred_main), dim=1).cpu().data[0].numpy()*cfg.TEST.LAMBDA_MAIN
                if cfg.TEST.LAMBDA_AUX != 0:
                    output_ += F.softmax(interp(pred_aux), dim=1).cpu().data[0].numpy()*cfg.TEST.LAMBDA_AUX
                if output is None:
                    output = model_weight * output_
                else:
                    output += model_weight * output_
                if cfg.TEST.FLIP:
                    image_flip = torch.flip(image,[3])
                    pred_aux_flip, pred_main_flip, _ = model(image_flip.cuda(device))
                    
                    output_flip_ = F.softmax(interp(pred_main_flip), dim=1).cpu().data[0].numpy()*cfg.TEST.LAMBDA_MAIN
                    if cfg.TEST.LAMBDA_AUX != 0:
                        output_flip_ += F.softmax(interp(pred_aux_flip), dim=1).cpu().data[0].numpy()*cfg.TEST.LAMBDA_AUX
                    if output_flip is None:
                        output_flip = model_weight * output_flip_
                    else:
                        output_flip += model_weight * output_flip_
            
            assert output is not None, 'Output is None'
            if cfg.TEST.FLIP:
                output = (output + output_flip[:,:,::-1] )/2
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
    
        label = label.numpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
        if index > 0 and index % 100 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(
                index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
    inters_over_union_classes = per_class_iu(hist)
    print(f'mIoU = \t{round(np.nanmean(inters_over_union_classes) * 100, 2)}')
    if verbose:
        display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes, hist)

def visualize(cfg, models,
                device, test_loader, interp,
                fixed_test_size, verbose):
    assert len(cfg.TEST.RESTORE_FROM) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.TEST.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    for index, batch in tqdm(enumerate(test_loader)):
        image, label, _, name = batch
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            output = None
            for model, model_weight in zip(models, cfg.TEST.MODEL_WEIGHT):
                pred_main = model(image.cuda(device))[1]
                output_ = interp(pred_main).cpu().data[0].numpy()
                if output is None:
                    output = model_weight * output_
                else:
                    output += model_weight * output_
            assert output is not None, 'Output is None'
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
        label = label.numpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
        #grid_image = make_grid(image[0].clone().cpu().data, 3, normalize=True)
        grid_prediction = make_grid(torch.from_numpy(color_mapping(np.asarray(output, dtype=np.uint8)).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
        grid_gt = make_grid(torch.from_numpy(color_mapping(label).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
        grid = make_grid([grid_gt, grid_prediction])
        show(grid, name)
    inters_over_union_classes = per_class_iu(hist)
    print(f'mIoU = \t{round(np.nanmean(inters_over_union_classes) * 100, 2)}')

def eval_best(cfg, models,
              device, test_loader, interp,
              fixed_test_size, verbose, flip=False, scales=[]):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    start_iter = cfg.TEST.SNAPSHOT_STEP
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    perfs =[] 
    cache_path = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'all_res' + str(cfg.TEST.INPUT_SIZE_TARGET) + '.pkl')
    if osp.exists(cache_path):
        all_res = pickle_load(cache_path)
    else:
        all_res = {}
    if cfg.TRAIN.ORDER == "Marginal":
        sort_order =[0, 13, 8, 2, 10, 11, 17,12, 14, 9, 4, 1, 6,18,3,5,7,15,16] 
    elif cfg.TRAIN.ORDER == "Pre_defined":
        sort_order = [8, 10, 13, 2, 0, 11, 12, 6, 15, 17, 9, 5, 18, 3, 4, 1, 16, 14, 7]
    elif cfg.TRAIN.ORDER == "Borders":
        sort_order = [0, 2, 10, 1, 8, 13, 9, 3, 14, 5, 4, 15, 11, 6, 7, 16, 17, 12, 18]
    else:
        sort_order = [2, 8, 0, 5, 10, 1, 9, 3, 13, 11, 4, 6, 14, 7, 15, 17, 12, 16, 18] 
    cur_best_miou = -1
    cur_best_model = ''
    evolution = [0 for _ in range(19)]  
    figures = [plt.figure(i) for i in range(19)] 
    for i_iter in range(start_iter, max_iter + 1, step):
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{i_iter}.pth')
        if not osp.exists(restore_from):
            # continue
            if cfg.TEST.WAIT_MODEL:
                print('Waiting for model..!')
                while not osp.exists(restore_from):
                    time.sleep(5)
        print("Evaluating model", restore_from)
        if i_iter not in all_res.keys():
            load_checkpoint_for_evaluation(models[0], restore_from, device)
            # eval
            hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            # for index, batch in enumerate(test_loader):
            #     image, _, _, name = batch
            test_iter = iter(test_loader)
            #interps = [nn.Upsample(size=(1024*scale, 2048*scale), mode='bilinear', align_corners=True) for scale in scales] 
            for index in tqdm(range(len(test_loader))):
                image, label, _, name = next(test_iter)
                
                if not fixed_test_size:
                    interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
                with torch.no_grad():
                    pred_main = models[0](image.cuda(device))[1]
                    
                    output = F.softmax(interp(pred_main), dim=1).cpu().data[0].numpy()
                    output = output.transpose(1, 2, 0)
                    output = np.argmax(output, axis=2)
                label = label.numpy()[0]
                hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
                if verbose and index > 0 and index % 100 == 0:
                    print('{:d} / {:d}: {:0.2f}'.format(
                        index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
            inters_over_union_classes = per_class_iu(hist)
            all_res[i_iter] = {"ious": inters_over_union_classes,
                               "hist": hist} 
            pickle_dump(all_res, cache_path)
        else:
            inters_over_union_classes = all_res[i_iter]["ious"]
            hist =  all_res[i_iter]["hist"]
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        if cur_best_miou < computed_miou:
            cur_best_miou = computed_miou
            cur_best_model = restore_from
        perfs.append(computed_miou)
        print('\tCurrent mIoU:', computed_miou)
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best mIoU:', cur_best_miou)
        if verbose:
            for i in range(19):
                plt.figure(i)
                plt.xticks(range(19), test_loader.dataset.class_names[sort_order], rotation = 45)
                current = round(hist[i, i]/hist[i, :].sum() * 100)
                plt.scatter(i_iter+1, current - evolution[i])
                evolution[i] = current 

            display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes, hist)
    print(perfs)

def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    try:
        model.load_state_dict(saved_state_dict)
    except:
        new_params = model.state_dict().copy()
        if 'feature_extractor' in saved_state_dict:
            feat = saved_state_dict['feature_extractor']

            for i in feat:
                translate = i[len("modlue.backbone."):] 
                new_params[translate] = feat[i] 

            clas = saved_state_dict['classifier'] 

            for i in clas:
                translate = i[len("modlue."):] 
                new_params["layer6." + translate] = clas[i] 
        elif 'state_dict' in saved_state_dict:
            
            for i in saved_state_dict['state_dict']:
                new_params[i[7:]]  = saved_state_dict['state_dict'][i] 
        else:
            for i in saved_state_dict:
                new_params[i[7:]]= saved_state_dict[i] 
            model.load_state_dict(new_params)
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes, hist):
    ious =[] 
    for ind_class in range(cfg.NUM_CLASSES):
        ious.append(inters_over_union_classes[ind_class])
        print(name_classes[ind_class]
              + '\t' + str(round(ious[-1] * 100, 2)) + '\t' + str(round(hist[ind_class, ind_class]/hist[ind_class, :].sum() * 100, 2))) 
    print(ious)
    #hist = hist/np.sum(hist, 0)
    """
    print('\t'+'\t'.join(name_classes))
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class]
              + '\t' + '\t'.join(['{:.2E}'.format(a) for a in hist[ind_class, :]]))
    """
