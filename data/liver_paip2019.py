import os
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw

import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import shutil
import random
    
class LiverC(data.Dataset):
    """
    General Histo dataset:
    input: root path of images, list to indicate indexes of slide
    output: a PIL Image and label
    """

    CLASSES = (
        "N",
        "T",
        "U"
    )

    def __init__(self, img_root, anno_root = None, data_aug=True, prob=0.5, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, root_dir/slides/images.
            slides (list): list of indexing slide
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_root = img_root
        self.anno_root = anno_root
        cls = LiverC.CLASSES
        self.category2id = dict(zip(cls, range(len(cls))))
        self.id2category = dict(zip(range(len(cls)), cls))
        
        self.data_aug = data_aug
        self.prob = prob
        self.transform = transform

    def __len__(self):
        return len(self.walk_root_dir()[0])

    def __getitem__(self, idx):
        image, mask, label = self.pull_item(idx)
        if self.data_aug:
            cj = T.ColorJitter(0.2,0.2,0.1,0.1)
            image = cj(image)
            
            if random.random() < self.prob:
                image = F.hflip(image)
                mask = F.hflip(mask)
                
            if random.random() < self.prob:
                image = F.vflip(image)
                mask = F.vflip(mask)

            if random.random() < self.prob:
                degrees = [0,90,180,270]
                degree_idx = random.randint(0,3)
                degree = degrees[degree_idx]
                RR = T.RandomRotation((degree,degree))
                image = RR(image)
                mask = RR(mask)
        
        if self.transform is not None:
            image = self.transform(image)
            # ToTensor() is not suitable for multi-label mask
#             mask = self.transform(mask)
            if np.array(mask).max()==255:
                mask = torch.from_numpy(np.array(mask)/255).type(torch.long)
            else:
                mask = torch.from_numpy(np.array(mask)).type(torch.long)
        
        return image, (mask, label)
    
    def walk_root_dir(self):
        images=[]
        masks = []
        labels=[]
        for dirpath, subdirs, files in os.walk(self.img_root):
            for x in files:
                # only U label selected or remove U
#                 if x.endswith(('.png','.jpg')) and x.split('_')[0] == 'U':
                if x.endswith(('.png','.jpg')):
                    images.append(os.path.join(dirpath, x))
                    maskx = x.split('.')[0]+'_viable_mask.bmp'
                    masks.append(os.path.join(dirpath, maskx))
                    if x.split('_')[0] == 'U':
                        labels.append('T') # index 0 for labels
                    else:
                        labels.append(x.split('_')[0]) # index 0 for labels
        return images, masks, labels
    
    def pull_item(self, idx):
        """
        Args:
            index (int): idx
        Returns:
            tuple: Tuple (image, mask, label)
        """
        # load image or mask as a PIL Image
        images, masks, labels = self.walk_root_dir()
        img_name = images[idx]
        image = Image.open(img_name).convert("RGB")
        mask_name = masks[idx]
        mask = Image.open(mask_name)
        label_name = labels[idx]
        label = self.category2id[label_name]
        return image, mask, label
    
    def statistic(self):
        images, masks, labels = self.walk_root_dir()
        label_dict = {}
        for label in labels:
            if label not in label_dict.keys():
                label_dict[label]=1
            else:
                label_dict[label]+=1
        return label_dict