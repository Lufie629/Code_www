# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            # HJQ CODE ==================================
            # print("HJQ CODE ==================================")
            # print(trains)
            # print('END CODE ==================================')

            # TextRCNN
            if config.model == 'TextRCNN':
                outputs,_ = model(trains)

            # TextCNN TextRNN FastText DPCNN TextRNN-Att
            else:
                outputs = model(trains)

            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    probs_all = np.array([], dtype=float)  # 添加一个数组来保存概率值
    feature_all = []
    with torch.no_grad():
        for texts, labels in data_iter:
            start_time = time.time()
            # TextRCNN
            if config.model == 'TextRCNN':
                outputs, feature = model(texts)
                for i in feature.data.cpu().numpy():
                    feature_all.append(i)
            else:
                outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            # predict_all = np.append(predict_all, predic)

            ## 
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 获取属于正类别的概率
            probs_all = np.append(probs_all, probs)
            end_time = time.time()
        


    # if test:
    #     # 只测试RCNN
    #     import json

    #     print(type(labels_all))
    #     print(type(feature_all))

    #     labels_all = labels_all.tolist()
    #     tmp = []
    #     for f in feature_all:
    #         tmp.append(f.tolist())
    #         # print(type(f))
    #     data = {
    #         'label': labels_all,
    #         'user_features': tmp
    #     }
    #     # Convert the data to JSON


    #     json_data = json.dumps(data)

    #     # Save JSON data to a file
    #     with open('data.json', 'w') as json_file:
    #         json_file.write(json_data)
    #     print(len(feature_all))
    #     print(len(labels_all))
    # with open('roc.txt', 'a') as file:
    #     # 将内容写入文件
    #     file.write("M\n")
    #     for i in np.array(labels_all):
    #         file.write(str(i)+" ")
    #     file.write("\n")
    #     for i in probs_all:
    #         file.write(str(i)+" ")
    #     file.write("\n")

    acc_ = metrics.accuracy_score(labels_all, predict_all)
    precision = metrics.precision_score(labels_all, predict_all)
    recall = metrics.recall_score(labels_all, predict_all)
    f1 = metrics.f1_score(labels_all, predict_all)
    auc = metrics.roc_auc_score(labels_all, probs_all)
    print("auc:", auc)
    print("acc:", acc_)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)
    # print(report)
    # print(confusion)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

# def evaluate(config, model, data_iter, test=False):
#     model.eval()
#     loss_total = 0
#     predict_all = np.array([], dtype=int)
#     labels_all = np.array([], dtype=int)
#     probs_all = np.array([], dtype=float)  # 添加一个数组来保存概率值
#     with torch.no_grad():
#         for texts, labels in data_iter:
#             outputs = model(texts)
#             loss = F.cross_entropy(outputs, labels)
#             loss_total += loss
#             labels = labels.data.cpu().numpy()
#             predic = torch.max(outputs.data, 1)[1].cpu().numpy()
#             labels_all = np.append(labels_all, labels)
#             predict_all = np.append(predict_all, predic)
#             probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 获取属于正类别的概率
#             probs_all = np.append(probs_all, probs)
#     acc = metrics.accuracy_score(labels_all, predict_all)
#     if test:
#         report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
#         confusion = metrics.confusion_matrix(labels_all, predict_all)
#         auc = roc_auc_score(labels_all, probs_all)  # 计算AUC
#         fpr, tpr, thresholds = roc_curve(labels_all, probs_all)  # 计算ROC曲线的参数
#         print("auc:",auc)
#         # 绘制ROC曲线
#         # plt.figure(figsize=(8, 8))
#         # plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(auc))
#         # plt.plot([0, 1], [0, 1], 'k--')
#         # plt.xlim([0.0, 1.0])
#         # plt.ylim([0.0, 1.05])
#         # plt.xlabel('False Positive Rate')
#         # plt.ylabel('True Positive Rate')
#         # plt.title('Receiver Operating Characteristic (ROC)')
#         # plt.legend(loc='lower right')
#         # plt.show()
#         return acc, loss_total / len(data_iter), report, confusion
#     return acc, loss_total / len(data_iter)
# from sklearn import metrics

# def evaluate(config, model, data_iter, test=False):
#     model.eval()
#     loss_total = 0
#     predict_all = np.array([], dtype=int)
#     labels_all = np.array([], dtype=int)
#     probs_all = np.array([], dtype=float)
    
#     with torch.no_grad():
#         for texts, labels in data_iter:
#             outputs = model(texts)
#             loss = F.cross_entropy(outputs, labels)
#             loss_total += loss.item()
#             labels = labels.data.cpu().numpy()
#             predic = torch.max(outputs.data, 1)[1].cpu().numpy()
#             labels_all = np.append(labels_all, labels)
#             predict_all = np.append(predict_all, predic)
#             probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
#             probs_all = np.append(probs_all, probs)
    
#     acc = metrics.accuracy_score(labels_all, predict_all)
#     precision = metrics.precision_score(labels_all, predict_all)
#     recall = metrics.recall_score(labels_all, predict_all)
#     f1 = metrics.f1_score(labels_all, predict_all)
#     auc = metrics.roc_auc_score(labels_all, probs_all)
    
#     if test:
#         report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
#         confusion = metrics.confusion_matrix(labels_all, predict_all)
#         print("auc:", auc)
#         print("acc:", acc)
#         print("precision:", precision)
#         print("recall:", recall)
#         print("f1:", f1)
#         print(report)
#         print(confusion)
    
#     return acc, loss_total / len(data_iter)


# from sklearn import metrics

# def evaluate(config, model, data_iter, test=False):
#     model.eval()
#     loss_total = 0
#     predict_all = np.array([], dtype=int)
#     labels_all = np.array([], dtype=int)
#     probs_all = np.array([], dtype=float)  # 添加一个数组来保存概率值
    
#     with torch.no_grad():
#         for texts, labels in data_iter:
#             outputs = model(texts)
#             loss = F.cross_entropy(outputs, labels)
#             loss_total += loss.item()
#             labels = labels.data.cpu().numpy()
#             predic = torch.max(outputs.data, 1)[1].cpu().numpy()
#             labels_all = np.append(labels_all, labels)
#             predict_all = np.append(predict_all, predic)
#             probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 获取属于正类别的概率
#             probs_all = np.append(probs_all, probs)
    
#     acc = metrics.accuracy_score(labels_all, predict_all)
#     precision = metrics.precision_score(labels_all, predict_all)
#     recall = metrics.recall_score(labels_all, predict_all)
#     f1 = metrics.f1_score(labels_all, predict_all)
#     auc = metrics.roc_auc_score(labels_all, probs_all)
    
#     if test:
#         report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
#         confusion = metrics.confusion_matrix(labels_all, predict_all)
#         print("auc:", auc)
#         print("acc:", acc)
#         print("precision:", precision)
#         print("recall:", recall)
#         print("f1:", f1)
#         print(report)
#         print(confusion)
    
#     return acc, loss_total / len(data_iter)