import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def reduce_act_on_non_included(predicted, included, mask):
    n, c, h, w = predicted.size()
    b = torch.zeros(c).cuda()
    b[included] += 1 
    ent = -torch.log(predicted.softmax(1).view(-1, c)[mask.repeat(1,c)].view(mask.sum().item(), c))
    prob = ent*b
    return prob.sum()/(c*n*h*w)

def cross_entropy(pred, target, weights=None, norm=False):
    oh_labels = nn.functional.one_hot(target, 19)
    pre_loss = -oh_labels*F.log_softmax(pred, dim=1)#/(oh_labels.sum(0) + 1)
    
    class_loss = torch.sum(pre_loss, dim=0)
    mask = pre_loss != 0
    if norm:
        class_loss = class_loss/(torch.sum(mask, dim=0) + 1e-30)
        return torch.mean(weights*class_loss), class_loss
    if weights is None:
        per_clas = class_loss/(torch.sum(mask, dim=0) + 1e-30)
        return torch.sum(class_loss)/((mask).shape[0]), per_clas
    return torch.sum(weights*class_loss)/((mask).shape[0]) , class_loss/(torch.sum(mask, dim=0) + 1e-30)  
    
    
    

def cross_entropy_2d(predict, target, included=None, weights=None):
    """
    Args:
        predict:(n, c, h, w)
        target:(n, h, w)
    """
    assert not target.requires_grad
    assert predict.dim() == 4
    if target.dim() != 3:
        target = target.squeeze()[None,:] 
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
    assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
    
    n, c, h, w = predict.size()
    target_mask = (target != 255)*(target >= 0)
    if included is not None:
        included_mask = False
        for i in included:
            included_mask = (target == i)| included_mask 
    else:
        included_mask = True
    target_mask = target_mask & included_mask
    target = target[target_mask]
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    #print(target)
    if not target_mask.any():
        return cross_entropy(predict, target, weights)
        loss = F.cross_entropy(predict, torch.zeros(target.shape).long().cuda(), size_average=True)
    else:
        return cross_entropy(predict, target, weights)
        loss = F.cross_entropy(predict, target, size_average=True, weight=weights)
    return loss 


def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))


class IW_MaxSquareloss(nn.Module):
    def __init__(self, ignore_index= 255, num_class=19, ratio=0.2):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
        self.ratio = ratio
    
    def forward(self, pred, prob, label=None):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :param label(optional): the map for counting label numbers (N, C, H, W)
        :return: maximum squares loss with image-wise weighting factor
        """
        # prob -= 0.5
        
        
        N, C, H, W = prob.size()
        mask = (prob != self.ignore_index)
        maxpred, argpred = torch.max(prob, 1)
        mask_arg = (maxpred != self.ignore_index)
        argpred = torch.where(mask_arg, argpred, torch.ones(1).to(prob.device, dtype=torch.long)*self.ignore_index)
        if label is None:
            label = argpred
        else:
            #REMOVE 
            mask = (label == 255)+(label < 0)
            label[mask]  = 0
            label = nn.functional.one_hot(label, 19)
        
        weights = []
        batch_size = prob.size(0)
        for i in range(batch_size):
            hist = torch.histc(label[i].cpu().data.float(), 
                            bins=self.num_class+1, min=-1,
                            max=self.num_class-1).float()
            hist = hist[1:]
            weight = (1/torch.max(torch.pow(hist, self.ratio)*torch.pow(hist.sum(), 1-self.ratio), torch.ones(1))).to(argpred.device)[argpred[i]].detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0).repeat(1,19, 1, 1).reshape(prob.shape)
        mask = mask_arg.unsqueeze(1).expand_as(prob)
        prior = torch.mean(prob, (2,3), True).detach()
        #print(weights.size(), prob.size())
        loss = -torch.sum((torch.pow(prob, 2)*weights)[mask]) / (batch_size*self.num_class*torch.sum(weights))
        return loss

class MaxSquareloss(nn.Module):
    def __init__(self, ignore_index= -1, num_class=19):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
    
    def forward(self, pred, prob):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :return: maximum squares loss
        """
        # prob -= 0.5
        mask = (prob != 255)*(prob >= 0)
        mask = (prob != self.ignore_index)    
        loss = -torch.mean(torch.pow(prob, 2)[mask]) / 2
        return loss


def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    #print(soft_label, pred)
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))