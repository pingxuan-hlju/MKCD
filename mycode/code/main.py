import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import model as Model
from data_loader import MyDataset

device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def train(model,cost,optimizer,trainSet,testSet,features,epoch,cross, scheduler,device):
    features=features.float().to(device)  
    isSave=0      
    for i in range(epoch):
        running_loss = 0.0 
        model.train()  
        for x1,x2,label in trainSet:
            x1,x2,label=x1.long().to(device),x2.long().to(device),label.long().to(device) 
            out=model(x1,x2,features,device) 
            loss=cost(out,label)        
            optimizer.zero_grad()      
            loss.backward()             
            optimizer.step()            
            running_loss += loss.item()
        print(f"Epoch {i+1}, Loss: {running_loss}")
        if i+1==epoch:
            isSave=1
            print('epoch:%d'%(i+1))           
            tacc(model,trainSet,features,0,isSave,cross,device)     
            tacc(model,testSet,features,1,isSave,cross,device)        
        scheduler.step()
        torch.cuda.empty_cache()


def evaluate(model, cost, testSet, features):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x1,x2,label in testSet:
            x1,x2,label=x1.long().to(device),x2.long().to(device),label.long().to(device) 
            out=model(x1,x2,features) 
            loss=cost(out,label)       
            val_loss += loss.item()
    return val_loss / len(testSet)


def tacc(model,tset,fea,string,s,cros,device):
    correct=0   
    total=0   
    st={0:'train_acc',1:'test_acc'}
    predall,yall=torch.tensor([]).to(device),torch.tensor([]).to(device)
    model.eval()  
    for x1,x2,y in tset:
        x1,x2,y=x1.long().to(device),x2.long().to(device),y.long().to(device)
        pred=model(x1,x2,fea,device).data   
        if s==1:
            predall=torch.cat([predall,torch.as_tensor(pred)],dim=0)
            yall=torch.cat([yall,torch.as_tensor(y)])
        a=torch.max(pred,1)[1]
        total+=y.size(0)
        correct+=(a==y).sum() 
    if string==1 and s==1:
        torch.save((predall,yall),'./circ_CNNplt_%d'%cros)
    print(st[string]+str((correct/total).item()))

    
if __name__ == "__main__":
    _, cd, features, trainSet_index, testSet_index = torch.load('circ_CNN.pth') 
    print(cd.shape, features[0].shape, trainSet_index[0].shape, testSet_index[0].shape)
    
    learn_rate=1e-3  
    epoch=40         
    batch=32        

    for i in range(5):
        net=Model.Net().to(device) 
        cost=nn.CrossEntropyLoss()    
        optimizer=torch.optim.AdamW(net.parameters(),learn_rate,weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        trainSet=DataLoader(MyDataset(trainSet_index[i],cd),batch,shuffle=True)   
        testSet=DataLoader(MyDataset(testSet_index[i],cd),batch,shuffle=False)  
        train(net,cost,optimizer,trainSet,testSet,features[i],epoch,i, scheduler,device)  