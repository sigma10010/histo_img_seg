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

class FIVES(data.Dataset):
    """
    General Histo dataset:
    input: root path of images, list to indicate indexes of slide
    output: a PIL Image and label
    """
    CLASSES = (
        "__background__ ",
        "vessel",
        
    )

    LABEL_DIC = {'A':0, 'D':1, 'G':2, 'N':3}

    def __init__(self, img_root, anno_root, data_aug=True, prob=0.5, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, root_dir/slides/images.
            slides (list): list of indexing slide
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_root = img_root
        self.anno_root = anno_root
        # self.img_size = img_size
        cls = FIVES.CLASSES
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
            mask = self.transform(mask)
        
        return image, (mask, label)

    
    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        image, _ = self.pull_image(idx)
        img_width, img_height = image.size
        return {"height": img_height, "width": img_width}
    def map_class_id_to_class_name(self, class_id):
        return FIVES.CLASSES[class_id]
    
    def toImg(self, x):
        t=transforms.ToPILImage()
        img = t(x)
        return img
        
    
    def walk_root_dir(self):
        names=[]
        wholePathes=[]
        annoNames=[]
        annoWholePathes=[]
        for dirpath, subdirs, files in os.walk(self.img_root):
            for x in files:
                if x.endswith(('.png','.jpg', '.jpeg', '.tif')):
                    names.append(x)
                    wholePathes.append(os.path.join(dirpath, x))
                    annoWholePathes.append(os.path.join(self.anno_root, x))
        return names, wholePathes, names, annoWholePathes
    
    
    def get_mask(self, idx, anno):
        width = self.get_img_info(idx)['width']
        height = self.get_img_info(idx)['height']
        img = Image.new('1', (width, height), 0) # pil 1 bit image
        for polygon in anno['polygons']:
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        return img 
        
    def get_groundtruth(self, index):
        image, anno = self.pull_item(index)
        labels = torch.from_numpy(np.array(anno['labels']))
        difficults = torch.tensor(anno['difficults'])
        # create a BoxList from the boxes
        # image need to be a PIL Image
        boxlist = BoxList(anno['bndboxes'], image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)
        boxlist.add_field("difficult", difficults)
        return boxlist
        
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            

        images = torch.stack(images, dim=0)

        return images, boxes  # tensor (N, 3, 300, 300), 3 lists of N tensors each
    
    def pull_item(self, idx):
        """
        Args:
            index (int): idx
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        # load the image as a PIL Image
        imgNames, imgWholePathes, annoNames, annoWholePathes = self.walk_root_dir()

        img_name = imgWholePathes[idx]
        image = Image.open(img_name).convert("RGB")

        mask_name = annoWholePathes[idx]
        mask = Image.open(mask_name).convert("L")
        mask = mask.point(lambda x: 255 if x>100 else 0).convert('1')

        label_str = imgNames[idx].split('_')[1]
        label = np.array(FIVES.LABEL_DIC[label_str])

        return image, mask, label
        
    def pull_image(self, idx):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            numpy img
        '''
        imgNames, imgWholePathes, annoNames, annoWholePathes = self.walk_root_dir()
        img_name = imgWholePathes[idx]
        img=io.imread(img_name)
        img = np.array(img)
        return img

    def pull_anno(self, idx):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [xmin, ymin, xmax, ymax] # interger
        '''
        imgNames, imgWholePathes, annoNames, annoWholePathes = self.walk_root_dir()
        anno_name = annoWholePathes[idx]
        anno = self.paser_xml(anno_name)
        return anno['bndboxes'], imgNames[idx]
    
    def statistic(self):
        num_cell = 0
        for i in range(self.__len__()):
            num_cell += len(self.get_groundtruth(i))
        return num_cell
