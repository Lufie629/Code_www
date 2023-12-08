import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/bot/bot/my_behavior_train.txt'                                # 训练集
        self.dev_path = dataset + '/data/bot/bot/my_behavior_val.txt'                                    # 验证集
        self.test_path = dataset + '/data/bot/bot/my_behavior_test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/bot/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/bot/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '_64bbot.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(             
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda:0')   # 设备

        self.dropout = 0.1                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 30                                            # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.pad_size = 100                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 30           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.num_filters * len(config.filter_sizes), 64)
        self.fc2 = nn.Linear(64, config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        # print(out[1])
        # print(out.shape)
        # out = self.fc(out)
        
        out = self.fc1(out)
        out = self.dropout(out)
        # with open('./bot_embedding.txt','a+',encoding='utf-8') as f:
        #     f.write(str(out[0].data)+'\n')
        #     f.write(str(out[1].data)+'\n')
        out = self.fc2(out)
        # print(out)
        return out