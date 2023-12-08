import pandas as pd
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,roc_curve
from torch.utils.data import random_split
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
from sklearn.ensemble import RandomForestClassifier
import time
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 


def main(argv):
    if argv[1] == '--datasets':
        try:
            name = argv[2]
            return name
        except:
            return "Wrong command!"
    else:
        return "Wrong command!"

def load_data(args):
    args.path='../../Datasets_process/Data/detptData/'
    with open(os.path.join(args.path,'vocab.json'), 'r') as f:
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
if __name__ == '__main__':
    global args, pred_test, pred_test_prob, label_test
    args = parser.parse_args()
    data, x, label = load_data(args)

    X = x.numpy()
    y = label.numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=76)
    rf.fit(X_train, y_train)
    start_time = time.time()
    y_pred = rf.predict(X_test)
    end_time = time.time()
 
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
 
 
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_ = auc(fpr, tpr)
    confusion_matrix = confusion_matrix(y_test, y_pred)

    print('acc:', acc)
    print('precision:', precision)
    print('recall:', recall)
    print('f1score:', f1score)
    print('mcc:', mcc)
    print('auc:', auc_)
    print('confusion_matrix:\n', confusion_matrix)
