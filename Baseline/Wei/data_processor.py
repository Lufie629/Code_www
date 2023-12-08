# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:19:01 2020
读取数据并对数据做预处理
统计出训练数据中出现频次最多的5k个单词，用这出现最多的5k个单词创建词表（词向量）
对于测试数据，直接用训练数据构建的词表
@author: 
"""
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import csv
import pandas as pd
import numpy as np
import sklearn
import json
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from os import listdir
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.nn import functional


glove_path = r"/data2/whr/zqy/glove.twitter.27B.25d.txt"
torch.manual_seed(100)

from sklearn.model_selection import train_test_split
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
    dec_outputs=torch.LongTensor(dec_outputs) 
    print("loading features...")
    cat_features = torch.load(args.path + "det_new_cat_properties_tensor.pt", map_location="cpu")
    prop_features = torch.load(args.path + "det_new_num_properties_tensor.pt", map_location="cpu")
    des_features = torch.load(args.path + "det_new_des_tensor.pt", map_location="cpu")
    enc_inputs = torch.ones(14062,1).long()

    
   
    x = torch.cat((cat_features, prop_features, des_features,dec_inputs,enc_inputs,tweets_tensor), dim=1)
    
    print(" CODE ==================================")
    print("cat_features shape:   ", cat_features.shape)
    print("prop_features shape:  ", prop_features.shape)
    print("dec_inputs shape:     ", dec_inputs.shape)
    print("enc_inputs shape:     ", enc_inputs.shape)
    print("tweet_tensor shape: ", tweets_tensor.shape)
    print("des_features shape:   ", des_features.shape)
    print("x shape: ", x.shape)
    print(" END load_data.py=======================================")
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


parser = argparse.ArgumentParser(description="Reproduction of Heterogeneity-aware Bot detection with Relational Graph Transformers")
parser.add_argument("--path", type=str, default="../../Datasets_process/Data/detptData/", help="dataset path")
parser.add_argument("--numeric_num", type=int, default=6, help="numeric_num")
parser.add_argument("--linear_channels", type=int, default=30, help="linear channels")
parser.add_argument("--cat_num", type=int, default=4, help="catgorical features")

parser.add_argument("--dec_inputs_num", type=int, default=101, help="dec_inputs features")
parser.add_argument("--tweet_embeddings_num", type=int, default=77568, help="weet_embeddings_num")
parser.add_argument("--dec_outputs_num", type=int, default=101, help="dec_outputs features")
parser.add_argument("--enc_inputs_num", type=int, default=1, help="enc_inputs features")

parser.add_argument("--des_channel", type=int, default=768, help="description channel")
parser.add_argument("--tweet_channel", type=int, default=768, help="tweet channel")
parser.add_argument("--out_channel", type=int, default=30, help="out channel")
parser.add_argument("--dropout", type=float, default=0.3, help="dropout channel")
parser.add_argument("--trans_head", type=int, default=2, help="trans_head")
parser.add_argument("--semantic_head", type=int, default=2, help="semantic_head")
parser.add_argument("--batch_size", type=int, default=10, help="batch_size")
parser.add_argument("--epochs", type=int, default=35, help="epochs")
parser.add_argument("--lr", type=float, default=5e-4, help="lr")
parser.add_argument("--l2_reg", type=float, default=1e-5, help="l2_reg")
parser.add_argument("--random_seed", type=int, default=None, help="random")
parser.add_argument("--test_batch_size", type=int, default=10, help="test_batchsize")

class DataProcessor(object):
    def read_text(self,is_train_data):
        #读取原始文本数据
        #is_train_data==True表示读取训练数据
        #is_train_data==False表示读取测试数据
        datas = []
        labels = []

        """
        if(is_train_data):
            #训练数据目录
            pos_path = "./datasets/aclImdb/train/pos/" 
            neg_path = "./datasets/aclImdb/train/neg/" 
        else:
            #测试数据目录
            pos_path = "./datasets/aclImdb/test/pos/" 
            neg_path = "./datasets/aclImdb/test/neg/"
        pos_files= os.listdir(pos_path)  #获取文件夹下的所有文件名称
        neg_files = os.listdir(neg_path)
        
        for file_name in pos_files: #遍历文件夹
            file_position = pos_path + file_name
            with open(file_position, "r",encoding='utf-8') as f:  #打开文件
                data = f.read()   #读取文件
                datas.append(data)
                labels.append([1,0]) #正类标签维[1,0]
        
        for file_name in neg_files:
            file_position = neg_path + file_name 
            with open(file_position, "r",encoding='utf-8') as f:
                data = f.read()
                datas.append(data)
                labels.append([0,1]) #负类标签维[0,1]
        """
        if (is_train_data):
            json_file = pd.read_json('./cresci-20151.json')
            csv_file = sklearn.utils.shuffle(json_file)  # 随机打乱
            for line in csv_file.itertuples():
                if (getattr(line, 'split') == "train") & (getattr(line, 'label') == "human"):
                    datas.append(getattr(line, 'text'))
                    labels.append([1, 0])  #负类标签维[1,0]
                elif (getattr(line, 'split') == "train") & (getattr(line, 'label') == "bot"):
                    datas.append(getattr(line, 'text'))
                    labels.append([0,1])  # 正类标签维[0,1]


        else:
            json_file = pd.read_json('./cresci-20151.json')
            csv_file = sklearn.utils.shuffle(json_file)  # 随机打乱
            for line in csv_file.itertuples():
                if (getattr(line, 'split') == "test") & (getattr(line, 'label') == "human"):
                    datas.append(getattr(line, 'text'))
                    labels.append([1, 0])   #负类标签维[1,0]
                elif (getattr(line, 'split') == "test") & (getattr(line, 'label') == "bot"):
                    datas.append(getattr(line, 'text'))
                    labels.append([0,1]) # 正类标签维[0,1]


        return datas, labels

    def word_count(self, datas):
        #统计单词出现的频次，并将其降序排列，得出出现频次最多的单词
        dic = {}
        for data in datas:
            data_list = data.split()
            for word in data_list:
                word = word.lower() #所有单词转化为小写
                if(word in dic):
                    dic[word] += 1
                else:
                    dic[word] = 1
        word_count_sorted = sorted(dic.items(), key=lambda item:item[1], reverse=True)
        return  word_count_sorted

    def word_index(self, datas, vocab_size):
        #创建词表
        word_count_sorted = self.word_count(datas)
        word2index = {}
        #词表中未出现的词
        word2index["<unk>"] = 0
        #句子添加的padding
        word2index["<pad>"] = 1

        #词表的实际大小由词的数量和限定大小决定
        vocab_size = min(len(word_count_sorted), vocab_size)
        for i in range(vocab_size):
            word = word_count_sorted[i][0]
            word2index[word] = i + 2

        return word2index, vocab_size

    def get_datasets(self, vocab_size, embedding_size, max_len):
        #注，由于nn.Embedding每次生成的词嵌入不固定，因此此处同时获取训练数据的词嵌入和测试数据的词嵌入
        #测试数据的词表也用训练数据创建
        # train_datas, train_labels = self.read_text(is_train_data=True)

        # word2index, vocab_size = self.word_index(train_datas, vocab_size)

        # test_datas, test_labels = self.read_text(is_train_data = False)

        # train_features = []
        # ###添加glove
        # embedding_index = dict()

        # with open(glove_path, 'r', encoding='utf-8') as f:
        #     line = f.readline()
        #     while line:
        #         values = line.split()
        #         word = values[0]
        #         coefs = np.asarray(values[1:])
        #         embedding_index[word] = coefs
        #         line = f.readline()
        # embedding_matrix = np.zeros((vocab_size + 2, embedding_size))
        # for word, i in word2index.items():
        #     embedding_vector = embedding_index.get(word)
        #     if embedding_vector is not None:
        #         embedding_matrix[i] = embedding_vector


        # for data in train_datas:
        #     feature = []
        #     data_list = data.split()
        #     for word in data_list:
        #         word = word.lower() #词表中的单词均为小写
        #         if word in word2index:
        #             feature.append(word2index[word])
        #         else:
        #             feature.append(word2index["<unk>"]) #词表中未出现的词用<unk>代替
        #         if(len(feature)==max_len): #限制句子的最大长度，超出部分直接截断
        #             break
        #     #对未达到最大长度的句子添加padding
        #     feature = feature + [word2index["<pad>"]] * (max_len - len(feature))
        #     train_features.append(feature)

        # test_features = []
        # for data in test_datas:
        #     feature = []
        #     data_list = data.split()
        #     for word in data_list:
        #         word = word.lower() #词表中的单词均为小写
        #         if word in word2index:
        #             feature.append(word2index[word])
        #         else:
        #             feature.append(word2index["<unk>"]) #词表中未出现的词用<unk>代替
        #         if(len(feature)==max_len): #限制句子的最大长度，超出部分直接截断
        #             break
        #     #对未达到最大长度的句子添加padding
        #     feature = feature + [word2index["<pad>"]] * (max_len - len(feature))
        #     test_features.append(feature)

        #将词的index转换成tensor,train_features中数据的维度需要一致，否则会报错
        # train_features = torch.LongTensor(train_features)
        # train_labels = torch.FloatTensor(train_labels)

        # test_features = torch.LongTensor(test_features)
        # test_labels = torch.FloatTensor(test_labels)

        # #将词转化为embedding
        # #词表中有两个特殊的词<unk>和<pad>，所以词表实际大小为vocab_size + 2
        # embed = nn.Embedding(vocab_size + 2, embedding_size)
        # embed.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix))
        # embed.weight.requires_grad = False
        # train_features = embed(train_features)
        # test_features = embed(test_features)

        # 将词变为float
        # train_features = torch.FloatTensor(train_features)
        # test_features = torch.FloatTensor(test_features)

       

        args = parser.parse_args()
        data, x, label = load_data(args)

          # 先将数据和标签转换成numpy数组
        X = x.numpy()
        y = label.numpy()

        # 划分数据集成训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
        
        return train_datasets, test_datasets
