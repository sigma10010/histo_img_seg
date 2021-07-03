# Selective-scale networks

import torch
from torch import nn


class SS(nn.Module):
    def __init__(self, M, ch_in, r):
        """ 
        Args:
            M (int): number of heads.
            ch_in (int): input channel dimensionality.
            r: the radio for compute d, the length of z.
        """
        super(SS, self).__init__()
        d = int(ch_in/r)
        self.M = M
        self.fc = nn.Linear(ch_in, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, ch_in)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, features):
        '''list of feature maps, len: M
        '''
        for i, fea in enumerate(features):
            fea = fea.unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1) # [b,M,features,h,w]
        fea_U = torch.sum(feas, dim=1) # [b,ch_in,h,w]
        fea_s = fea_U.mean(-1).mean(-1) # [b,ch_in]
        fea_z = self.fc(fea_s) # [b,d]
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1) # [b,1,ch_in]
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors) # [b,M,ch_in]
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1) # [b,M,ch_in,1,1]
        fea_v = (feas * attention_vectors).sum(dim=1) # [b,ch_in,h,w], M attention
        return fea_v