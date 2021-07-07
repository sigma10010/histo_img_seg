import argparse
import os

from datasets.data_split import divide_data

import numpy as np
import torch.nn.functional as F
from torchvision import transforms

from skimage import io
import os
import openslide as ops
from PIL import Image
from metrics.evaluation import *

from models.u_net import U_Net, AttU_Net, XXU_Net
from models.unet import UNet, UNet_V1, UNet_V2, UNet_V3, UNet_V4 # new version

import torch
import csv

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def wsi_prediction(s, model, foreground, level, n_class=2, patch_size = 400, step = 200, foreground_thres = 0.25, t=transforms.ToTensor()):
    step = patch_size//2
    
    whole_size_i = s.level_dimensions[level][1] # y
    whole_size_j = s.level_dimensions[level][0] # x

    patch_size_i = patch_size
    patch_size_j = patch_size

    result = np.zeros((n_class,whole_size_i,whole_size_j))
    for i in range(0, whole_size_i, step):
        for j in range(0, whole_size_j, step):
            if i+patch_size_i>whole_size_i:
                i = whole_size_i-patch_size_i
            if j+patch_size_j>whole_size_j:
                j = whole_size_j-patch_size_j

            tem_j=int(j*(s.level_downsamples[level]/s.level_downsamples[1]))
            tem_i=int(i*(s.level_downsamples[level]/s.level_downsamples[1]))
            tem_size=int(patch_size*(s.level_downsamples[level]/s.level_downsamples[1]))
            sub_foreground=foreground[tem_i:tem_i+tem_size, tem_j:tem_j+tem_size]
            if sub_foreground.sum()/(tem_size*tem_size)<=foreground_thres:
                continue

            img = s.read_region((int(j*s.level_downsamples[level]),int(i*s.level_downsamples[level])),level,(patch_size,patch_size))
            img = t(img)[0:3,:,:].unsqueeze(0)
            SR, CR = model(img)
            SR = SR.squeeze().detach().numpy()

            if i==0 and j==0:
                result[:,i:i+patch_size_i,j:j+patch_size_j] = SR
            elif i==0 and j>0:
                overlay1 = result[:,i:i+patch_size_i,j:j+patch_size_j-step]
                overlay2 = SR[:,:,0:patch_size_j-step]
                SR[:,:,0:patch_size_j-step] = (overlay1+overlay2)/2
                result[:,i:i+patch_size_i,j:j+patch_size_j] = SR
            elif j==0 and i>0:
                overlay1 = result[:,i:i+patch_size_i-step,j:j+patch_size_j]
                overlay2 = SR[:,0:patch_size_j-step,:]
                SR[:,0:patch_size_j-step,:] = (overlay1+overlay2)/2
                result[:,i:i+patch_size_i,j:j+patch_size_j] = SR
            else:
                overlay1 = result[:,i:i+patch_size_i-step,j:j+patch_size_j]
                overlay2 = SR[:,0:patch_size_j-step,:]
                SR[:,0:patch_size_j-step,:] = (overlay1+overlay2)/2

                overlay3 = result[:,i+patch_size_i-step:i+patch_size_i,j:j+patch_size_j-step]
                overlay4 = SR[:,patch_size_i-step:patch_size_i,0:patch_size_j-step]
                SR[:,patch_size_i-step:patch_size_i,0:patch_size_j-step] = (overlay3+overlay4)/2

                result[:,i:i+patch_size_i,j:j+patch_size_j] = SR
    return result

def test_wsi(unet, svsf, p1, level, patch_size = 400, category = 1, t=transforms.ToTensor()):
    '''
    svsf: (list) wsi to validate
    '''
    Image.MAX_IMAGE_PIXELS = 933120000000
    
    score = 0
    jss = []
    for i, f in enumerate(svsf):
        s=ops.open_slide(p1+f)
        image= s.read_region((0,0),2,s.level_dimensions[2])
        foreground=io.imread(p1+f.split('.')[0]+'.tif')
        pm = p1+f.split('.')[0]+'_viable_level%d.png'%level
        mask = Image.open(pm)
        result = wsi_prediction(s, unet, foreground, level, patch_size = patch_size, foreground_thres = 0.25)
        
        tep_js = get_MultiClassJS(result,t(mask),label=category)
        jss.append(tep_js)
        if tep_js>0.65:
            score+=tep_js
            
        tep_dc = get_MultiClassDC(result,t(mask),label=category)
        
        print('%d/%d'%(i+1,len(svsf)), tep_js, tep_dc)
    return (score/len(svsf)), jss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=400)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--category', type=int, default=1, help='category for evaluation label')
    parser.add_argument('--t', type=int, default=2, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    parser.add_argument('--reduction_ratio', type=int, default=None, help='reduction ratio for attention layer') 
    parser.add_argument('--n_skip', type=int, default=4, help='number of skip-connection layers, <= depth-1')
    parser.add_argument('--n_head', type=int, default=1, help='number of heads for prediction, 1 <= depth-1') 
    parser.add_argument('--att_mode', type=str, default='cbam', help='cbam/bam/se')
    parser.add_argument('--conv_type', type=str, default='basic', help='basic/sk')
    parser.add_argument('--is_shortcut', type=str2bool, default=False)
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.5)
    
    # log
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--val_step', type=int, default=2)
    
    # loss
    parser.add_argument('--loss_type', type=str, default='nll', help='l1/nll/focal/iou/dice/multitask/multiloss')
    parser.add_argument('--alpha', type=float, default=1)        # alpha for l1 loss
    parser.add_argument('--gamma', type=float, default=1)        # gamma for l1/Focal loss
    parser.add_argument('--balance', type=list, default=None)   # balance factor

    # misc
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model_type', type=str, default='MHU_Net', help='MHU_Net/U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='/mnt/DATA_OTHER/paip2019/results/checkpoint/seg/')
    parser.add_argument('--root_path', type=str, default='/mnt/DATA_OTHER/paip2019/patches/seg/level1_400/')
    parser.add_argument('--train_path', type=str, default='/mnt/DATA_OTHER/paip2019/patches/seg/level1_400/train/')
    parser.add_argument('--train_anno_path', type=str, default=None)
    parser.add_argument('--valid_path', type=str, default='/mnt/DATA_OTHER/paip2019/patches/seg/level1_400/validation/')
    parser.add_argument('--wsi_path', type=str, default='/mnt/DATA_OTHER/paip2019/original/')
    parser.add_argument('--valid_anno_path', type=str, default=None)
    parser.add_argument('--test_path', type=str, default='./fundus_images/test/')
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--fold', type=int, default=3, help='5-fold cross validation')
    parser.add_argument('--level', type=int, default=2, help='1/2')
    
    # other
    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    config.root_path = '/mnt/DATA_OTHER/paip2019/patches/seg/level%d_%d/'%(config.level, config.image_size)
    config.train_path = '/mnt/DATA_OTHER/paip2019/patches/seg/level%d_%d/train/'%(config.level, config.image_size)
    config.valid_path = '/mnt/DATA_OTHER/paip2019/patches/seg/level%d_%d/validation/'%(config.level, config.image_size)
    if config.n_skip>config.depth-1:
        config.n_skip = config.depth-1
    
    # divide data for k-fold cross validation
    dd = divide_data(config.root_path)
    dd.reset() # move data to all/
    dd.divide(config.fold)
    
    """Build model"""
    unet = None
    if config.model_type =='U_Net':
        unet = U_Net(img_ch=3, n_classes = config.n_classes, activation = torch.nn.Softmax(dim=1))
    elif config.model_type =='R2U_Net':
        unet = R2U_Net(img_ch=3,t=config.t, n_classes = config.n_classes, activation = torch.nn.Softmax(dim=1))
    elif config.model_type =='AttU_Net':
        unet = AttU_Net(img_ch=3, n_classes = config.n_classes, activation = torch.nn.Softmax(dim=1))
    elif config.model_type == 'R2AttU_Net':
        unet = R2AttU_Net(img_ch=3,t=config.t, n_classes = config.n_classes, activation = torch.nn.Softmax(dim=1))
    elif config.model_type in ['UNet', 'SEU_Net', 'CBAMU_Net', 'BAMU_Net']:
        unet = UNet(img_ch=3, n_classes=config.n_classes, init_features=config.width, network_depth=config.depth, reduction_ratio=config.reduction_ratio, att_mode = config.att_mode, activation = torch.nn.Softmax(dim=1))
    elif config.model_type in ['SKU_Net', 'SK-SC-U_Net', 'SK-SE-U_Net']:
        unet = UNet_V1(reduction_ratio=config.reduction_ratio, att_mode = config.att_mode, is_shortcut = config.is_shortcut, conv_type = config.conv_type, activation = torch.nn.Softmax(dim=1))
    elif config.model_type == 'MHU_Net':
        unet = UNet_V2(img_ch=3, n_classes=config.n_classes, n_head = config.n_head, is_head_selective = False, is_shortcut = False, activation = torch.nn.Softmax(dim=1))
    elif config.model_type == 'SCU_Net':
        unet = UNet_V3(img_ch=3, n_classes=config.n_classes, n_head = config.n_head, is_scale_selective = False, is_shortcut = True, activation = torch.nn.Softmax(dim=1))
    elif config.model_type in ['SSU_Net', 'SK-SSU_Net', 'SE-SSU_Net', 'SC-SSU_Net' ]:
        unet = UNet_V4(reduction_ratio=config.reduction_ratio, n_head = config.n_head, att_mode = config.att_mode, is_scale_selective = True, is_shortcut = config.is_shortcut, conv_type = config.conv_type, activation = torch.nn.Softmax(dim=1))
    else:
        raise NotImplementedError(config.model_type+" is not implemented")

        
    unet_path = os.path.join(config.model_path, '%s-%s-level%s-size%s-depth%s-width%s-n_classes%s-alpha%s-gamma%s-nhead%s-fold%s.pkl'%(config.model_type, config.loss_type, config.level, config.image_size, config.depth, config.width, config.n_classes, config.alpha, config.gamma, config.n_head, config.fold))
    print('try to load weights from: %s'%unet_path)
    unet.load_state_dict(torch.load(unet_path))
    unet.train(False)
    unet.eval()
    
#     print(unet)
    
    # validation slide
    valf=[]
    for f in os.listdir(config.valid_path):
        valf.append(f)
    print(valf)
        
    svsf=[]
    for f in os.listdir(config.wsi_path):
        if f.endswith(('.svs','.SVS')) and f.split('.')[0] in valf:
            svsf.append(f)
    print(svsf)
    
    score, jss = test_wsi(unet, svsf, p1 = config.wsi_path, level = config.level, patch_size = config.image_size, category = config.category, t=transforms.ToTensor())
    print(score)
    f = open(os.path.join(config.result_path,'result_wsi.csv'), 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow([config.model_type, config.loss_type, config.level, config.image_size, config.depth, config.width, config.n_classes, config.alpha, config.gamma, config.n_head, config.fold, score]+[js for js in jss])
    f.close()