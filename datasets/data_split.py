import os
import numpy as np
import shutil
import random

class divide_data():
    '''
    divide data by slide level
    root_dir: parent root of all/
    rate: train:val:test
    '''
    def __init__(self, root_dir, rate=[1,1,1]):
        self.rate = rate  # train:val:test
        self.root = root_dir
        self.root_train = self.root+'train/'
        self.root_test = self.root+'test/'
        self.root_val = self.root+'validation/'
        self.root_all = self.root+'all/'
        if not os.path.exists(self.root_train):
            os.mkdir(self.root_train)
        if not os.path.exists(self.root_test):
            os.mkdir(self.root_test)
        if not os.path.exists(self.root_val):
            os.mkdir(self.root_val)

    def reset(self):
        for f in os.listdir(self.root_train):
            shutil.move(self.root_train+f,self.root_all)
        for f in os.listdir(self.root_test):
            shutil.move(self.root_test+f,self.root_all)
        for f in os.listdir(self.root_val):
            shutil.move(self.root_val+f,self.root_all)
        return 'sucess'

    def divide(self, fold, k = 5):
        fname = os.path.join(self.root,'experiments','index.npy')
        if os.path.isfile(fname):
            index = np.load(fname)
        else:
            index=list(range(len(os.listdir(self.root_all))))
            random.shuffle(index)
            index = np.array(index)
            np.save(fname, index)
            
        dirs=[]
        n_fold = len(index)//k
        if fold == k and (len(index)%k)>0:
            val_indx = list(index[(fold-1)*n_fold:(fold-1)*n_fold+n_fold]) + list(index[-(len(index)%k):])
        else:
            val_indx = list(index[(fold-1)*n_fold:(fold-1)*n_fold+n_fold])
        for _, f in enumerate(os.listdir(self.root_all)):
            dirs.append(f)
        
        for i in range(len(dirs)):
            if i in val_indx:
                shutil.move(self.root_all+dirs[i],self.root_val)
            else:
                shutil.move(self.root_all+dirs[i],self.root_train)
        return 'sucess'