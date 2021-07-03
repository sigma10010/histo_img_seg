import torch
import numpy as np

# SR : Segmentation Result
# GT : Ground Truth

class Metric:
    def __init__(self):
        pass

    def __call__(self, SR, GT, threshold=0.5):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class MultiClassAccumulatedAccuracyMetric(Metric):
    """
    
    """

    def __init__(self):
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def __call__(self, SR,GT, label = 2):
        if SR.dim()>2:
            SR = SR.view(SR.size(0),SR.size(1),-1)  # N,C,H,W => N,C,H*W
            SR = SR.transpose(1,2)    # N,C,H*W => N,H*W,C
            SR = SR.contiguous().view(-1,SR.size(2))   # N,H*W,C => N*H*W,C
        GT = GT.view(-1,1) # N,H,W => N*H*W,1 or N, => N,1
        SR = SR.argmax(1,keepdim = True).type(torch.uint8)
        
        TP = ((SR==label)&(GT==label))
        FN = ((SR!=label)&(GT==label))
        FP = ((SR==label)&(GT!=label))
        TN = ((SR!=label)&(GT!=label))
    
        self.tp += torch.sum(TP)
        self.tn += torch.sum(TN)
        self.fp += torch.sum(FP)
        self.fn += torch.sum(FN)
        
        return self.value()

    def reset(self):
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def value(self):
        return 100.0 * (self.tp+self.tn) / (self.tp+self.tn+self.fp+self.fn)

    def name(self):
        return 'Accuracy'
    
class AccumulatedAccuracyMetric(Metric):
    """
    
    """

    def __init__(self):
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def __call__(self, SR,GT,threshold=0.5):
        SR = SR > threshold
        GT = GT == 1.0

        TP = ((SR==1)&(GT==1))
        FN = ((SR==0)&(GT==1))
        FP = ((SR==1)&(GT==0))
        TN = ((SR==0)&(GT==0))
    
        self.tp += torch.sum(TP)
        self.tn += torch.sum(TN)
        self.fp += torch.sum(FP)
        self.fn += torch.sum(FN)
        
        return self.value()

    def reset(self):
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def value(self):
        return 100.0 * (self.tp+self.tn) / (self.tp+self.tn+self.fp+self.fn)

    def name(self):
        return 'Accuracy'
    
class AccumulatedF1Metric(Metric):
    """
    F1 = DC : Dice Coefficient
    """

    def __init__(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def __call__(self, SR,GT,threshold=0.5):
        SR = SR > threshold
        GT = GT == 1.0

        TP = ((SR==1)&(GT==1))
        FN = ((SR==0)&(GT==1))
        FP = ((SR==1)&(GT==0))
    
        self.tp += torch.sum(TP)
        self.fp += torch.sum(FP)
        self.fn += torch.sum(FN)
        
        return self.value()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def value(self):
        return 100.0 * (2.0*self.tp) / (2.0*self.tp+self.fp+self.fn)

    def name(self):
        return 'F1'

class AccumulatedDCMetric(Metric):
    """
    F1 = DC : Dice Coefficient
    """

    def __init__(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def __call__(self, SR,GT,threshold=0.5):
        SR = SR > threshold
        GT = GT == 1.0

        TP = ((SR==1)&(GT==1))
        FN = ((SR==0)&(GT==1))
        FP = ((SR==1)&(GT==0))
    
        self.tp += torch.sum(TP)
        self.fp += torch.sum(FP)
        self.fn += torch.sum(FN)
        
        return self.value()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def value(self):
        return 100.0 * (2.0*self.tp) / (2.0*self.tp+self.fp+self.fn)

    def name(self):
        return 'DC'
    
class MultiClassAccumulatedDCMetric(Metric):
    """
    F1 = DC : Dice Coefficient
    """

    def __init__(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def __call__(self, SR,GT,label=2):
        if SR.dim()>2:
            SR = SR.view(SR.size(0),SR.size(1),-1)  # N,C,H,W => N,C,H*W
            SR = SR.transpose(1,2)    # N,C,H*W => N,H*W,C
            SR = SR.contiguous().view(-1,SR.size(2))   # N,H*W,C => N*H*W,C
        GT = GT.view(-1,1) # N,H,W => N*H*W,1 or N, => N,1
        SR = SR.argmax(1,keepdim = True).type(torch.uint8)
        
        TP = ((SR==label)&(GT==label))
        FN = ((SR!=label)&(GT==label))
        FP = ((SR==label)&(GT!=label))
    
        self.tp += torch.sum(TP)
        self.fp += torch.sum(FP)
        self.fn += torch.sum(FN)
        
        return self.value()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def value(self):
        return 100.0 * (2.0*self.tp) / (2.0*self.tp+self.fp+self.fn)

    def name(self):
        return 'DC'
    
class AccumulatedJSMetric(Metric):
    """
    JS = IOU: Jaccard Similarity
    """

    def __init__(self):
        self.inter = 0.0
        self.union = 0.0
        

    def __call__(self, SR,GT,threshold=0.5):
        SR = SR > threshold
        GT = GT == 1.0

        Inter = torch.sum((SR==1)&(GT==1)) # TP
        Union = torch.sum((SR==1)|(GT==1))
    
        self.inter += Inter
        self.union += Union
        
        return self.value()

    def reset(self):
        self.inter = 0.0
        self.union = 0.0
        

    def value(self):
        return 100.0 * (self.inter) / (self.union)

    def name(self):
        return 'JS'
    
class MultiClassAccumulatedJSMetric(Metric):
    """
    JS = IOU: Jaccard Similarity
    """

    def __init__(self):
        self.inter = 0.0
        self.union = 0.0
        

    def __call__(self, SR,GT,label=2):
        if SR.dim()>2:
            SR = SR.view(SR.size(0),SR.size(1),-1)  # N,C,H,W => N,C,H*W
            SR = SR.transpose(1,2)    # N,C,H*W => N,H*W,C
            SR = SR.contiguous().view(-1,SR.size(2))   # N,H*W,C => N*H*W,C
        GT = GT.view(-1,1) # N,H,W => N*H*W,1 or N, => N,1
        SR = SR.argmax(1,keepdim = True).type(torch.uint8)

        Inter = torch.sum((SR==label)&(GT==label)) # TP
        Union = torch.sum((SR==label)|(GT==label))
    
        self.inter += Inter
        self.union += Union
        
        return self.value()

    def reset(self):
        self.inter = 0.0
        self.union = 0.0
        

    def value(self):
        return 100.0 * (self.inter) / (self.union)

    def name(self):
        return 'JS'

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def get_acc(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == 1.0
    
    TP = ((SR==1)&(GT==1))
    FN = ((SR==0)&(GT==1))
    FP = ((SR==1)&(GT==0))
    TN = ((SR==0)&(GT==0))
    AC = float(torch.sum(TP|TN))/(float(torch.sum(TP|FN|TN|FP)) + 1e-6)     

    return AC

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == 1.0

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1)&(GT==1))
    FN = ((SR==0)&(GT==1))

    SE = float(torch.sum(TP))/(float(torch.sum(TP|FN)) + 1e-6)     
    
    return SE

class MultiClassAccumulatedSEMetric(Metric):
    """
    # Sensitivity == Recall
    """

    def __init__(self):
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def __call__(self, SR,GT, label = 2):
        if SR.dim()>2:
            SR = SR.view(SR.size(0),SR.size(1),-1)  # N,C,H,W => N,C,H*W
            SR = SR.transpose(1,2)    # N,C,H*W => N,H*W,C
            SR = SR.contiguous().view(-1,SR.size(2))   # N,H*W,C => N*H*W,C
        GT = GT.view(-1,1) # N,H,W => N*H*W,1 or N, => N,1
        SR = SR.argmax(1,keepdim = True).type(torch.uint8)
        
        TP = ((SR==label)&(GT==label))
        FN = ((SR!=label)&(GT==label))
        FP = ((SR==label)&(GT!=label))
        TN = ((SR!=label)&(GT!=label))
    
        self.tp += torch.sum(TP)
        self.tn += torch.sum(TN)
        self.fp += torch.sum(FP)
        self.fn += torch.sum(FN)
        
        return self.value()

    def reset(self):
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def value(self):
        return 100.0 * (self.tp) / (self.tp+self.fn)

    def name(self):
        return 'SE'



def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == 1.0

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0)&(GT==0))
    FP = ((SR==1)&(GT==0))

    SP = float(torch.sum(TN))/(float(torch.sum(TN|FP)) + 1e-6)
    
    return SP

class MultiClassAccumulatedSPMetric(Metric):
    """
    # SP: specificity
    """

    def __init__(self):
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def __call__(self, SR,GT, label = 2):
        if SR.dim()>2:
            SR = SR.view(SR.size(0),SR.size(1),-1)  # N,C,H,W => N,C,H*W
            SR = SR.transpose(1,2)    # N,C,H*W => N,H*W,C
            SR = SR.contiguous().view(-1,SR.size(2))   # N,H*W,C => N*H*W,C
        GT = GT.view(-1,1) # N,H,W => N*H*W,1 or N, => N,1
        SR = SR.argmax(1,keepdim = True).type(torch.uint8)
        
        TP = ((SR==label)&(GT==label))
        FN = ((SR!=label)&(GT==label))
        FP = ((SR==label)&(GT!=label))
        TN = ((SR!=label)&(GT!=label))
    
        self.tp += torch.sum(TP)
        self.tn += torch.sum(TN)
        self.fp += torch.sum(FP)
        self.fn += torch.sum(FN)
        
        return self.value()

    def reset(self):
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def value(self):
        return 100.0 * (self.tn) / (self.tn+self.fp)

    def name(self):
        return 'SP'

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == 1.0

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1)&(GT==1))
    FP = ((SR==1)&(GT==0))

    PC = float(torch.sum(TP))/(float(torch.sum(TP|FP)) + 1e-6)

    return PC

class MultiClassAccumulatedPCMetric(Metric):
    """
    # PC: precision
    """

    def __init__(self):
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def __call__(self, SR,GT, label = 2):
        if SR.dim()>2:
            SR = SR.view(SR.size(0),SR.size(1),-1)  # N,C,H,W => N,C,H*W
            SR = SR.transpose(1,2)    # N,C,H*W => N,H*W,C
            SR = SR.contiguous().view(-1,SR.size(2))   # N,H*W,C => N*H*W,C
        GT = GT.view(-1,1) # N,H,W => N*H*W,1 or N, => N,1
        SR = SR.argmax(1,keepdim = True).type(torch.uint8)
        
        TP = ((SR==label)&(GT==label))
        FN = ((SR!=label)&(GT==label))
        FP = ((SR==label)&(GT!=label))
        TN = ((SR!=label)&(GT!=label))
    
        self.tp += torch.sum(TP)
        self.tn += torch.sum(TN)
        self.fp += torch.sum(FP)
        self.fn += torch.sum(FN)
        
        return self.value()

    def reset(self):
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def value(self):
        return 100.0 * (self.tp) / (self.tp+self.fp)

    def name(self):
        return 'PC'

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == 1.0
    
    Inter = torch.sum((SR==1)&(GT==1)) # TP
    Union = torch.sum((SR==1)|(GT==1))
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == 1.0
    
    TP = ((SR==1)&(GT==1))
    FN = ((SR==0)&(GT==1))
    FP = ((SR==1)&(GT==0))

    DC = 2.0*float(torch.sum(TP))/(2.0*float(torch.sum(TP))+float(torch.sum(FN|FP)) + 1e-6)

    return DC

def get_MultiClassJS(SR,GT,label=1):
    '''
    SR:numpy due to bug of argmax of torch
    '''
    # JS : Jaccard similarity
    SR = torch.from_numpy(SR.argmax(0)).type(torch.uint8).view(GT.size())
    
    Inter = torch.sum((SR==label)&(GT==label)) # TP
    Union = torch.sum((SR==label)|(GT==label))
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_MultiClassDC(SR,GT,label=1):
    # DC : Dice Coefficient
#     SR = SR.argmax(0,keepdim = True).type(torch.uint8)
    SR = torch.from_numpy(SR.argmax(0)).type(torch.uint8).view(GT.size())
        
    TP = ((SR==label)&(GT==label))
    FN = ((SR!=label)&(GT==label))
    FP = ((SR==label)&(GT!=label))

    DC = 2.0*float(torch.sum(TP))/(2.0*float(torch.sum(TP))+float(torch.sum(FN|FP)) + 1e-6)

    return DC



