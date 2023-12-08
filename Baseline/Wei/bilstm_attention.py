# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:25:58 2020
文本分类 双向LSTM + Attention 算法
@author: 
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from data_processor import DataProcessor
import numpy as np
torch.manual_seed(100) #保证每次运行初始化的随机数相同
import time
vocab_size = 50000   #词表大小
embedding_size = 78448   #词向量维度
num_classes = 2    #二分类
sentence_max_len = 64  #单个句子的长度
hidden_size = 16

num_layers =3  #三层lstm
num_directions = 2 #双向lstm
lr = 1e-3
batch_size = 16   
epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Bi-LSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, embedding_size,hidden_size, num_layers, num_directions, num_classes):
        super(BiLSTMModel, self).__init__()
        
        self.input_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        
        
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers = num_layers, bidirectional = (num_directions == 2))
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.liner = nn.Linear(hidden_size, num_classes)
        self.act_func = nn.Softmax(dim=1)
    
    def forward(self, x):
        # print(x.shape)
        #lstm的输入维度为 [seq_len, batch, input_size]
        #x [batch_size, sentence_length, embedding_size]
        x = x.unsqueeze(1)
        x = x.permute(1, 0, 2)         #[sentence_length, batch_size, embedding_size]
        
        #由于数据集不一定是预先设置的batch_size的整数倍，所以用size(1)获取当前数据实际的batch
        batch_size = x.size(1)
        
        #设置lstm最初的前项输出
        h_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        
        #out[seq_len, batch, num_directions * hidden_size]。多层lstm，out只保存最后一层每个时间步t的输出h_t
        #h_n, c_n [num_layers * num_directions, batch, hidden_size]
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        #将双向lstm的输出拆分为前向输出和后向输出
        (forward_out, backward_out) = torch.chunk(out, 2, dim = 2)
        out = forward_out + backward_out  #[seq_len, batch, hidden_size]
        out = out.permute(1, 0, 2)  #[batch, seq_len, hidden_size]
        
        #为了使用到lstm最后一个时间步时，每层lstm的表达，用h_n生成attention的权重
        h_n = h_n.permute(1, 0, 2)  #[batch, num_layers * num_directions,  hidden_size]
        h_n = torch.sum(h_n, dim=1) #[batch, 1,  hidden_size]
        h_n = h_n.squeeze(dim=1)  #[batch, hidden_size]
        
        attention_w = self.attention_weights_layer(h_n)  #[batch, hidden_size]
        attention_w = attention_w.unsqueeze(dim=1) #[batch, 1, hidden_size]
        
        attention_context = torch.bmm(attention_w, out.transpose(1, 2))  #[batch, 1, seq_len]
        softmax_w = F.softmax(attention_context, dim=-1)  #[batch, 1, seq_len],权重归一化
        
        x = torch.bmm(softmax_w, out)  #[batch, 1, hidden_size]
        x = x.squeeze(dim=1)  #[batch, hidden_size]
        feature = x
        x = self.liner(x)
        x = self.act_func(x)
        return x, feature
        
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
    sum_labels = []
    sum_preds = []
    sum_feature = []
    for datas, labels in test_loader:
        labels_cpu = labels
        
        datas = datas.to(device)
        labels = labels.to(device)
        
        
        preds,feature = model(datas)
        for i in feature.data.cpu().numpy():
                sum_feature.append(i.tolist())
        preds_ = preds

        for i in labels_cpu:
            sum_labels.append(i.item())

        labels_adjusted = torch.stack([1 - labels, labels], dim=1)
        loss = loss_func(preds, labels_adjusted)
        
        loss_val += loss.item() * datas.size(0)
        
        pred_probs = torch.softmax(preds, dim=1)
        positive_probs = pred_probs[:, 1].cpu()
        # print(pred_probs)
        for i in positive_probs:
            sum_preds.append(i.item())
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

    if epoch == 9:
        import json
        data = {
            'label': sum_labels,
            'user_features': sum_feature
        }
        # Convert the data to JSON


        json_data = json.dumps(data)

        # Save JSON data to a file
        with open('data_Wei.json', 'w') as json_file:
            json_file.write(json_data)
 

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
       

            preds,_ = model(datas)
            
            labels_adjusted = torch.stack([1 - labels, labels], dim=1)
            loss = loss_func(preds, labels_adjusted)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_val += loss.item() * datas.size(0)
            
            #获取预测的最大概率出现的位置
            # auc_preds = torch.cat(auc_preds,torch.softmax(preds, dim=1))
            # auc_labels= torch.cat(auc_labels,labels)

           
            pred_probs = torch.softmax(preds, dim=1)
            positive_probs = pred_probs[:, 1].cpu()
            if len(positive_probs) == batch_size:

                labels_ = labels.cpu()                
                auc = roc_auc_score(labels_.detach().numpy(), positive_probs.detach().numpy())
                auc_value.append(auc)
            
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

        print("Train Loss: {}, Train Acc: {}".format(train_loss,train_acc))
        print("Train Acc2: {}".format(train_acc2))
        print("Train auc: {}".format(sum(auc_value) / len(auc_value)))
        print("Train precision: {}, Train recall: {}".format(precision, recall))
        print("Train  F1: {}, positive rate: {}".format(F1, nump))

        start_time = time.time()
        test_acc = test(model, test_loader, loss_func,epoch)
        end_time = time.time()
        prediction_time = end_time - start_time
        print(f"times：{prediction_time}s")
        if(best_val_acc < test_acc):
            best_val_acc = test_acc
            best_model_params = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_params)
    return model

processor = DataProcessor()
train_datasets, test_datasets = processor.get_datasets(vocab_size=vocab_size, embedding_size=embedding_size, max_len=sentence_max_len)
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False)


model = BiLSTMModel(embedding_size, hidden_size, num_layers, num_directions, num_classes)

model = model.to(device)
tensor = torch.rand(1, 78448).to(device)
# start_time = time.time()
# model(tensor)
# end_time = time.time()
# print(end_time-start_time)
# print(tensor.shape)
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
# # 分析FLOPs
# flops = FlopCountAnalysis(model, tensor)
# print("FLOPs: ", flops.total())

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.BCELoss()
model = train(model, train_loader, test_loader, optimizer, loss_func, epochs)


