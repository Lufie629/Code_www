import argparse
import os
import os.path as osp
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import pandas
import torchvision
import json
import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv, GCNConv, Linear  # noqa
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn import preprocessing
import random
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pytorch_lightning as pl
from torch import nn
import torch
import json


# from Dataset import BotDataset
# from torch.utils.data import DataLoader
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from os import listdir
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.nn import functional
import os
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="Reproduction of Heterogeneity-aware Bot detection with Relational Graph Transformers")
parser.add_argument("--path", type=str, default="../../Datasets_process/Data/detptData", help="dataset path")
parser.add_argument("--numeric_num", type=int, default=6, help="dataset path")
parser.add_argument("--linear_channels", type=int, default=30, help="linear channels")
parser.add_argument("--cat_num", type=int, default=4, help="catgorical features")

parser.add_argument("--dec_inputs_num", type=int, default=101, help="dec_inputs features")
parser.add_argument("--tweet_embeddings_num", type=int, default=77568, help="weet_embeddings_num")
parser.add_argument("--dec_outputs_num", type=int, default=101, help="dec_outputs features")
parser.add_argument("--enc_inputs_num", type=int, default=1, help="enc_inputs features")

parser.add_argument("--des_channel", type=int, default=768, help="description channel")
parser.add_argument("--tweet_channel", type=int, default=768, help="tweet channel")
parser.add_argument("--out_channel", type=int, default=30, help="description channel")
parser.add_argument("--dropout", type=float, default=0.3, help="description channel")
parser.add_argument("--trans_head", type=int, default=2, help="description channel")
parser.add_argument("--semantic_head", type=int, default=2, help="description channel")
parser.add_argument("--batch_size", type=int, default=10, help="description channel")
parser.add_argument("--epochs", type=int, default=35, help="description channel")
parser.add_argument("--lr", type=float, default=5e-4, help="description channel")
parser.add_argument("--l2_reg", type=float, default=1e-5, help="description channel")
parser.add_argument("--random_seed", type=int, default=None, help="random")
parser.add_argument("--test_batch_size", type=int, default=10, help="random")




class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(5, 16, cached=True,
                             normalize=True)
        self.conv2 = GCNConv(16, 16, cached=True,
                             normalize=True)
        self.lin = Linear(78448,2)
        self.conv1 = ChebConv(78448, 16, K=2)
        self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self,x):
 
        x = self.lin(x)
        return F.softmax(x, dim=1)

def test(model, test_loader, loss_func, epoch):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    nump = 0.0
    auc_value = []
    sum_lables = []
    sum_preds = []
    for datas, labels in test_loader:
        datas = datas.to(device)
        labels = labels.to(device)
        for i in labels:
            sum_lables.append(i.item())
        preds = model(datas)
        
        for i in preds:
            sum_preds.append(i[1].item())
        labels_adjusted = torch.stack([1 - labels, labels], dim=1)
        loss = loss_func(preds, labels_adjusted)
        
        loss_val += loss.item() * datas.size(0)
        
        pred_probs = torch.softmax(preds, dim=1)
        positive_probs = pred_probs[:, 1].cpu()
        if len(positive_probs) == batch_size:

            labels_ = labels.cpu()                
            auc = roc_auc_score(labels_.detach().numpy(), positive_probs.detach().numpy())
            auc_value.append(auc)
        #获取预测的最大概率出现的位置
        preds = torch.argmax(preds, dim=1)
        labels = torch.argmax(labels_adjusted, dim=1)
        corrects += torch.sum(preds == labels).item()
        TP += torch.sum((preds == 1) & (labels == 1)).item()
        TN += torch.sum((preds == 0) & (labels == 0)).item()
        FP += torch.sum((preds == 1) & (labels == 0)).item()
        FN += torch.sum((preds == 0) & (labels == 1)).item()
        nump += torch.sum((preds == 1)).item()

    test_loss = loss_val / len(test_loader.dataset)
    test_acc = corrects / len(test_loader.dataset)
    test_acc2 = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    F1 = 2 * precision*recall / (precision + recall + 1e-6)
    nump = nump / len(test_loader.dataset)

    print("Test Loss: {}, Test Acc: {}".format(test_loss, test_acc))
    print("Test Acc2: {}".format(test_acc2))
    print("Test auc: {}".format(sum(auc_value) / len(auc_value)))
    print("Test precision: {}, Test recall: {}".format(precision, recall))
    print("Test  F1: {}, positive rate: {}".format(F1, nump))
    return test_acc

def train(model, train_loader,test_loader, optimizer, loss_func, epochs):
    best_val_acc = 0.0
    best_model_params = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        loss_val = 0.0
        corrects = 0.0
        TP = 0.0
        TN = 0.0
        FP = 0.0
        FN = 0.0
        nump = 0.0
        train_acc = 0.0
        train_acc2 = 0.0
        auc_value = []
        for datas, labels in train_loader:
            datas = datas.to(device)
            labels = labels.to(device)
            
            from thop import profile

            # Model
            # print('==> Building model..')
            # flops, params = profile(model, (datas,))
            # print('flops: ', flops, 'params: ', params)
            # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
            # exit(-1)
        
            preds = model(datas)
      
            labels_adjusted = torch.stack([1 - labels, labels], dim=1).to(device)
       
            loss = loss_func(preds, labels_adjusted)
       
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_val += loss.item() * datas.size(0)
            pred_probs = torch.softmax(preds, dim=1)
            positive_probs = pred_probs[:, 1].cpu()
       
            
            preds = torch.argmax(preds, dim=1)
            labels = torch.argmax(labels_adjusted, dim=1)
            
            corrects += torch.sum(preds == labels).item()
            TP += torch.sum((preds == 1) & (labels == 1)).item()
            TN += torch.sum((preds == 0) & (labels == 0)).item()
            FP += torch.sum((preds == 1) & (labels == 0)).item()
            FN += torch.sum((preds == 0) & (labels == 1)).item()
            nump += torch.sum((preds == 1)).item()

        train_loss = loss_val / len(train_loader.dataset)
        train_acc = corrects / len(train_loader.dataset)
        train_acc2 = (TP+TN)/(TP+TN+FP+FN+ 1e-6)
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        F1 = 2 * precision * recall / (precision + recall + 1e-6)
        nump = nump / len(train_loader.dataset)

        # print("=======")
        # print(torch.softmax())
 
        # auc = roc_auc_score(auc_labels, auc_preds)
        # print("Train auc: {}".format(auc))

        # print("Train Loss: {}, Train Acc: {}".format(train_loss,train_acc))
        # print("Train Acc2: {}".format(train_acc2))
        # print("Train auc: {}".format(sum(auc_value) / len(auc_value)))
        # print("Train precision: {}, Train recall: {}".format(precision, recall))
        # print("Train  F1: {}, positive rate: {}".format(F1, nump))
      
        test_acc = test(model, test_loader, loss_func,epoch)
      
   
        if(best_val_acc < test_acc):
            best_val_acc = test_acc
            best_model_params = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_params)
    return model


def load_data(args):
    args.path='../../Datasets_process/Data/detptData/'
    with open(args.path+'vocab.json', 'r') as f:
        vocab=json.load(f)
        vocab_size=len(vocab)

    pre_sentences=[]
    new_sentences=[]
    with open('../../Datasets_process/Data/detptData/det_behavior_data.txt', 'r',encoding='utf-8') as outfile:
        for b in outfile.readlines():
            tmp=b.strip().split('\t')[0]
            new_sentences.append(b.strip().split('\t')[0])
            pre_sentences.append(b.strip().split('\t')[0])
    sentences=[]
    for i in zip(pre_sentences[:7000],new_sentences[7062:]):
        sentences.append(i[0])
        sentences.append(i[1])
    sentences=sentences+pre_sentences[7000:7062]
            
    dec_inputs=[]
    dec_outputs=[]
    max_length=100
    tweets_tensor=torch.load(args.path+"det_new_tweets_tensor.pt",map_location="cpu")
    tweets_size,tweets_len,tweet_dim=tweets_tensor.shape
    end_tensor=torch.zeros(tweets_size,1,tweet_dim)
    tweets_tensor=torch.cat([tweets_tensor,end_tensor],1)
    tweets_tensor=torch.flatten(tweets_tensor,1,2)
    for sen in sentences:
        temp=[vocab[n] for n in sen.split(' ')]
        pad_length=len(temp)
        dec_input=[[vocab['<S>']]+temp[:100]+[vocab['<SP>']]*(max_length-pad_length)]
        dec_output=[temp[:100]+[vocab['<SP>']]*(max_length-pad_length)+[vocab['<E>']]]
        
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
        
    dec_inputs=torch.LongTensor(dec_inputs)
    # print(dec_inputs)
    dec_outputs=torch.LongTensor(dec_outputs) 
    # print(dec)
    print("loading features...")
    cat_features = torch.load(args.path + "det_new_cat_properties_tensor.pt", map_location="cpu")
    prop_features = torch.load(args.path + "det_new_num_properties_tensor.pt", map_location="cpu")
    des_features = torch.load(args.path + "det_new_des_tensor.pt", map_location="cpu")
    enc_inputs = torch.ones(14062,1).long()

    
   
    x = torch.cat((cat_features, prop_features, des_features,dec_inputs,enc_inputs,tweets_tensor), dim=1)
    

    print("cat_features shape:   ", cat_features.shape)
    print("prop_features shape:  ", prop_features.shape)
    print("dec_inputs shape:     ", dec_inputs.shape)
    print("enc_inputs shape:     ", enc_inputs.shape)
    print("tweet_tensor shape: ", tweets_tensor.shape)
    print("des_features shape:   ", des_features.shape)
    print("x shape: ", x.shape)

    
    # return
    print("loading edges & label...")
    edge_index = torch.load(args.path + "det_new_edge_index.pt", map_location="cpu")
    edge_type = torch.load(args.path + "det_new_edge_type.pt", map_location="cpu").unsqueeze(-1)
    label = torch.load(args.path + "det_new_label.pt", map_location="cpu")
    datalen=label.shape[0]
    data = Data(x=x, edge_index = edge_index, edge_attr=edge_type, y=label)

    
    print("loading index...")
    data.train_idx = torch.arange(0, int(datalen*0.7))
    data.valid_idx = torch.arange(int(datalen*0.7),int(datalen*0.9))
    data.test_idx = torch.arange(int(datalen*0.9), datalen)
    
    return data, x, label

if __name__ == '__main__':
    args = parser.parse_args()
    data, x, label = load_data(args)
     # 先将数据和标签转换成numpy数组
    X = x.numpy()
    y = label.numpy()

    # 划分数据集成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    #指定输入特征是否需要计算梯度
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    train_features = Variable(X_train, requires_grad=False)
    train_features = train_features.float()
    train_datasets = torch.utils.data.TensorDataset(train_features, y_train)

    test_features = Variable(X_test, requires_grad=False)
    test_features = test_features.float()
    test_datasets = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=16, shuffle=False)

    
  
    model = Net().to(device)
    batch_size = 16
    optimizer = torch.optim.Adam([dict(params=model.lin.parameters(), weight_decay=0)], lr=0.005)  # Only perform weight-decay on first convolution.

    loss_func = nn.BCELoss()
    model = train(model, train_loader, test_loader, optimizer, loss_func, 25)





