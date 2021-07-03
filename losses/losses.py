import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MultiTaskLoss(nn.Module):
    """
    multi-task loss: cls seg
    """

    def __init__(self, seg_loss, cls_loss):
        super(MultiTaskLoss, self).__init__()
        self.seg_loss = seg_loss
        self.cls_loss = cls_loss

    def forward(self, seg_output, mask, cls_output, label):
#         if seg_output is not None:
        seg_losses = self.seg_loss(seg_output, mask)
#         if cls_output is not None:
        cls_losses = self.cls_loss(cls_output, label)
        
        return seg_losses + cls_losses
    
class MultiLoss(nn.Module):
    """
    multi loss: cls seg
    """

    def __init__(self, losses = [], names = [], weights = []):
        super(MultiLoss, self).__init__()
        self.losses = losses
        self.names = names
        self.weights = weights

    def forward(self, seg_output, mask):
        loss_dict = {}
        total_loss = 0
        for i, loss in enumerate(self.losses):
            loss_value = loss(seg_output, mask)
            total_loss+=(loss_value*self.weights[i])
            key = self.names[i]
            loss_dict[key]=loss_value
        
        loss_dict['total_loss']=total_loss
        return loss_dict
    
class NLLLoss(nn.Module):
    def __init__(self, weight=None, size_average=None):
        super(NLLLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.criterion = torch.nn.NLLLoss(weight=self.weight, size_average=self.size_average)

    def forward(self, inputs, targets):
        inputs = torch.log(inputs)

        return self.criterion(inputs, targets)
    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)
#     print(IoU/b)
    return IoU/b

class IoULoss(torch.nn.Module):
    def __init__(self, label = 1, size_average = True):
        super(IoULoss, self).__init__()
        self.size_average = size_average
        self.category = label

    def forward(self, pred, target):
        target = target.unsqueeze(1) # N,1,H,W
        pred = pred[:,self.category,:,:].unsqueeze(1) # N,C,H,W -> N,1,H,W
        pred = torch.exp(pred) # convert log_softmax to softmax
        target = (target==self.category)

        return _iou(pred, target, self.size_average)
    
# class IoULoss(nn.Module):
#     def __init__(self, label = 1, weight=None, size_average=True):
#         super(IoULoss, self).__init__()
#         self.category = label

#     def forward(self, inputs, targets, smooth=1):
#         if inputs.dim()>2:
#             inputs = inputs.view(inputs.size(0),inputs.size(1),-1)  # N,C,H,W => N,C,H*W
#             inputs = inputs.transpose(1,2)    # N,C,H*W => N,H*W,C
#             inputs = inputs.contiguous().view(-1,inputs.size(2))   # N,H*W,C => N*H*W,C
#         targets = targets.view(-1,1)
#         inputs = inputs.gather(1,targets)
              
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         targets = (targets==self.category)
        
#         #intersection is equivalent to True Positive count
#         #union is the mutually inclusive area of all labels & predictions 
#         intersection = (inputs * targets).sum()
#         total = (inputs + targets).sum()
#         union = total - intersection 
        
#         IoU = (intersection + smooth)/(union + smooth)
        
#         return 1 - IoU

class GeneralizedL1Loss(nn.Module):
    '''
    l = at * \alpha * |p-p*|** \gamma
    input (Tensor): N,C or N,C,H,W
    beta: balance
    '''
    def __init__(self, alpha=1, gamma=1, beta = None, size_average=True):
        super(GeneralizedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        if isinstance(beta,(float,int)): self.beta = torch.Tensor([beta,1-beta])
        if isinstance(beta,list): self.beta = torch.Tensor(beta)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        p = input.gather(1,target)
        p = p.view(-1)
        if self.beta is not None:
            if self.beta.type()!=input.data.type():
                self.beta = self.beta.type_as(input.data)
            bt = self.beta.gather(0,target.data.view(-1))
            losses = Variable(bt) * self.alpha * (1-p)**self.gamma
            return losses.mean() if self.size_average else losses.sum()
        else:
            losses = self.alpha * (1-p)**self.gamma  # torch.abs(p - 1)
            return losses.mean() if self.size_average else losses.sum()

class FocalLoss(nn.Module):
    '''
    l = - \at * log pt * |1-pt|** \gamma 
    
    '''
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1) # N,H,W => N*H*W,1 or N, => N,1

#         logpt = F.log_softmax(input, dim=-1)
        logpt = torch.log(input)
        logpt = logpt.gather(1,target) # gather confidence of gt class based on target
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        losses = -1 * (1-pt)**self.gamma * logpt
        return losses.mean() if self.size_average else losses.sum()

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    output1/output2: embeddings nx2
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

class ContrastiveClassificationLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    output1/output2: embeddings nx2
    """

    def __init__(self, margin, c_loss):
        super(ContrastiveClassificationLoss, self).__init__()
        self.margin = margin
        self.c_loss = c_loss
        self.eps = 1e-9

    def forward(self, output1, output2, c_output1, target, label1, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        
#         class_losses = self.c_loss(torch.cat((c_output1, c_output2), 0), torch.cat((label1, label2), 0))
        class_losses = self.c_loss(c_output1, label1)
        
        return losses.mean()+class_losses if size_average else losses.sum()+class_losses

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
    
class TripletClassificationLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, c_loss):
        super(TripletClassificationLoss, self).__init__()
        self.margin = margin
        self.c_loss = c_loss

    def forward(self, anchor, positive, negative, c_output1, label, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        
        class_losses = self.c_loss(c_output1, label)
        
        return losses.mean()+class_losses if size_average else losses.sum()+class_losses

class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
