from model import BotRGCN
from Dataset import Twibot20
import torch
from torch import nn
from utils import accuracy,init_weights

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import matthews_corrcoef
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_size,dropout,lr,weight_decay=128,0.3,1e-3,5e-3

dataset=Twibot20(device=device,process=True,save=True)
des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()
# print(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx)
# print(num_prop.shape,category_prop.shape)
print('dataset loading down')


model=BotRGCN(num_prop_size=6,cat_prop_size=4,embedding_dimension=embedding_size).to(device)
loss=nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                    lr=lr,weight_decay=weight_decay)

def train(epoch):
    model.train()
    output, _ = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_train = loss(output[train_idx], labels[train_idx].long())
    acc_train = accuracy(output[train_idx], labels[train_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'acc_val: {:.4f}'.format(acc_val.item()),)
    return acc_train,loss_train

def test():
    model.eval()
    output, feature = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    preds_all = []
    labels_all = []
    feature_all = []
    preds = torch.softmax(output,dim=1)
    loss_test = loss(output[test_idx], labels[test_idx].long())
    acc_test = accuracy(output[test_idx], labels[test_idx])
    output=output.max(1)[1].to('cpu').detach().numpy()
    label=labels.to('cpu').detach().numpy()
    f1=f1_score(output[test_idx],label[test_idx])
    for i in feature[test_idx].data.cpu().numpy():
        # print(type(i.tolist()))
        feature_all.append(i.tolist())
    # print(output[test_idx],label[test_idx])
    pre=precision_score(output[test_idx],label[test_idx])
    rec=recall_score(output[test_idx],label[test_idx])
    acc=accuracy_score(output[test_idx],label[test_idx])
    auc=roc_auc_score(output[test_idx],label[test_idx])
    for l in label[test_idx]:
        labels_all.append(l.item())
    for p in output[test_idx]:
        preds_all.append(p.item())
    
    mcc=matthews_corrcoef(label[test_idx], output[test_idx])
    print("Test set results:",
            "test_loss= {:.4f}".format(loss_test.item()),
            "test_accuracy= {:.4f}".format(acc_test.item()),
            "test_accuracy= {:.4f}".format(acc.item()),
            "f1_score= {:.4f}".format(f1),
            "pre_score= {:.4f}".format(pre),
            "rec_score= {:.4f}".format(rec),
            "auc_score= {:.4f}".format(auc),
            
            # "mcc= {:.4f}".format(mcc.item()),
            "mcc= {:.4f}".format(mcc),
            )
    return acc_test,loss_test,f1_score

model.apply(init_weights)
epochs=100
for epoch in range(epochs):
    train(epoch)


# model= torch.load('/data/gluo/Bots_code/BotRGCN/botrgcn.pth',map_location='cuda:0')
test()

# torch.save(model,"botrgcn.pth")