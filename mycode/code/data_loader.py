import numpy as np
import itertools
import torch
from torch.utils.data import Dataset


def load_data(): 
    cd = np.load("mycode\data\circ_disease\circRNA_disease.npy")  
    dd = np.load("mycode\data\circ_disease\disease_disease.npy") 
    dm = np.load("mycode\data\circ_disease\disease_miRNA.npy") 
    cm = np.load("mycode\data\circ_disease\circRNA_miRNA.npy") 
    mm = np.load("mycode\data\circ_disease\miRNA_miRNA.npy")  
    return torch.tensor(cd), torch.tensor(dd), torch.tensor(dm), torch.tensor(cm), torch.tensor(mm)  


def calculate_sim(cd, dd): 
    s1 = cd.shape[0]  
    cc = torch.eye(s1) 
    m2 = dd * cd[:,None, :] 
    m1 = cd[:, :, None]
    for x, y in itertools.permutations(torch.linspace(0, s1 - 1, s1, dtype=torch.long),2): 
        x, y = x.item(), y.item() 
        m = m1[x, :, :] * m2[y, :, :]  
        if cd[x].sum() + cd[y].sum() == 0:
            cc[x, y] = 0  
        else:
            cc[x, y] = (m.max(dim=0, keepdim=True)[0].sum() +m.max(dim=1, keepdim=True)[0].sum()) / (cd[x].sum() + cd[y].sum())  
    return cc 


def split_dataset(assMatrix, dd, k, negr): 
    trainSet_index, testSet_index, cd_mask, cc = [], [], [], []
    rand_index = torch.randperm(assMatrix.sum().long().item()) 
    pos_index = torch.argwhere(assMatrix == 1).index_select(0, rand_index).T 
    neg_index = torch.argwhere(assMatrix == 0) 
    neg_index = neg_index.index_select(0, torch.randperm(neg_index.shape[0])).T 
    crossCount = int(pos_index.shape[1] / k) 
    for i in range(k):
        pos_Sample = torch.cat([pos_index[:, :(i * crossCount)], pos_index[:, ((i + 1) * crossCount):(k * crossCount)]],dim=1) 
        neg_Sample = torch.cat([neg_index[:, :(i * crossCount * negr)], neg_index[:,((i + 1) * crossCount * negr):(k * crossCount * negr)]],dim=1) 
        trainData = torch.cat([pos_Sample, neg_Sample], dim=1) 
        testData = torch.cat([pos_index[:, (i * crossCount):((i + 1) * crossCount)], neg_index[:, (k * crossCount * negr):]], dim=1) 
        trainSet_index.append(trainData)
        testSet_index.append(testData) 
        cdt = assMatrix.clone()
        cdt[pos_index[0, (i * crossCount):((i + 1) * crossCount)], pos_index[1, (i * crossCount):((i + 1) * crossCount)]] = 0 
        cd_mask.append(cdt)
        cc.append(calculate_sim(cdt, dd))
    return trainSet_index, testSet_index, cd_mask, cc 
  

def cfm(cc, cd, dd, dm, cm, mm):
    r1 = torch.cat([cc, cd, cm], dim=1)  
    r2 = torch.cat([cd.T, dd, dm], dim=1)
    r3 = torch.cat([cm.T, dm.T, mm], dim=1) 
    feature = torch.cat([r1, r2, r3], dim=0) 
    return feature


class MyDataset(Dataset):
    def __init__(self,tri,cd):
        self.tri=tri
        self.cd=cd
    def __getitem__(self,idx):
        x,y=self.tri[:,idx]
        label=self.cd[x][y]
        return x,y,label
    def __len__(self):
        return self.tri.shape[1]
   
    
# cd, dd, dm, cm, mm= load_data()
# k = 5
# neg_ratio = 1
# trainSet, testSet, cda, cc = split_dataset(cd, dd, k, neg_ratio)

# feas = []

# for i in range(k):
#     fea = cfm(cc[i], cd, dd, dm, cm, mm) 
#     feas.append(fea)

# torch.save([cd, feas, trainSet, testSet], 'circ_CNN.pth')