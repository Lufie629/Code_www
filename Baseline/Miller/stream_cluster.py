import os
import json
import ijson
import pandas as pd
import numpy as np
import argparse
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,roc_curve
import torch
import time


def main():
   
    num_cluster = 0
    k = 0
    epsilon = 0
    dbscan = False    
    
    k = 0.112
    num_cluster = 100
    dbscan = False
    print('loading......')
    path='../../Datasets_process/Data/detptData/'
    category_prop = torch.load(path + "det_new_cat_properties_tensor.pt", map_location="cpu")
    num_prop = torch.load(path + "det_new_num_properties_tensor.pt", map_location="cpu")
    X = torch.cat((category_prop,num_prop), dim=1).numpy()
    y = torch.load(path + "det_new_label.pt", map_location="cpu").numpy()
    datalen=y.shape[0]
    train_split=np.arange(datalen)<int(datalen*0.7)
    # train_split = torch.arange(0, int(datalen*0.7))
    # val_split = torch.arange(int(datalen*0.7),datalen)
    test_split = np.arange(datalen)>=int(datalen*0.7)
    test_set = np.where(np.arange(datalen)>=int(datalen*0.7))[0]
    val_split =  np.array([True]*datalen)^train_split^test_split
    num_user=datalen
    X_train_human = X[train_split | val_split][~y[train_split | val_split].astype(bool)]
    kmeans = KMeans(n_clusters=num_cluster, random_state=0)

    


    result = kmeans.fit(X_train_human)
    cluster = []
    for i in range(num_cluster):
        cluster.append(np.where(result.labels_ == i)[0])
    radius = []
    mean_dist = []
    for i in range(num_cluster):
        max_dist = 0
        sum_dist = 0
        for j in cluster[i]:
            max_dist = max(max_dist, np.linalg.norm(X_train_human[j] - result.cluster_centers_[i]))
            sum_dist += np.linalg.norm(X_train_human[j] - result.cluster_centers_[i])
        radius.append(max_dist)
        mean_dist.append(sum_dist / len(cluster[i]))
    eps = k * sum(radius) / num_cluster
    predict_y = []
    
    start_time = time.time()
    print(len(test_set))
    for i in range(len(test_set)):
        
        sign = 0
        for j in range(num_cluster):
            if np.linalg.norm(X[test_set[i]] - result.cluster_centers_[j]) < eps:
                sign = 1
        if sign == 1:
            predict_y.append(1)
        else:
            predict_y.append(0)
        end_time = time.time()
    prediction_time = end_time - start_time
    print(f"times：{prediction_time}s")
    predict = np.array(predict_y)
    predict_y_ = []
    for i in range(len(test_set)):
        sign = 0
        min_dist = float('inf')
        for j in range(num_cluster):
            dist = np.linalg.norm(X[test_set[i]] - result.cluster_centers_[j])
            if dist < eps:
                sign = 1
                prob = 1 - dist/eps
                min_dist = min(min_dist, prob)
        if sign == 1:
            predict_y_.append(min_dist)
        else:
            predict_y_.append(0)
   
   

    if dbscan:
        predict_y = np.array(predict_y)
        X_train_nd = np.vstack((X_train_human, X[np.array(test_set)[~predict_y.astype(bool)]]))
        dbs = DBSCAN(eps=epsilon, min_samples=num_user // 50)
        result_2 = dbs.fit(X_train_nd)
        predict_2 = result_2.fit_predict(X[np.array(test_set)[predict_y.astype(bool)]])
        map_1 = []
        for i in range(len(predict_y.astype(bool))):
            if predict_y.astype(bool)[i]:
                map_1.append(i)
        for i in range(len(predict_2)):
            if predict_2[i] >= 0:
                predict[map_1[i]] = 0
        
    print(f"acc: {accuracy_score(y[test_set], predict):.4f},recall: {recall_score(y[test_set], predict):.4f}, f1-score: {f1_score(y[test_set], predict):.4f}, roc_auc: {roc_auc_score(y[test_set], predict):.4f}")


if __name__ == '__main__':
    main()

