# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network,test
from importlib import import_module
import argparse
import os


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=False, default = 'TextRCNN', help='choose a model: TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()



if __name__ == '__main__':
    dataset = 'THUCNews' 
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    config.model = args.model
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    print(len(train_data),len(dev_data), len(test_data))
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    device=torch.device("cuda:0")
    config.n_vocab = len(vocab)
    model = x.Model(config).to(device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)


    
    train(config, model, train_iter, dev_iter, test_iter)
    # torch.save(model,args.model+'.pth')
    # 假设你的模型叫做 model，保存的权重文件是 "/data/gluo/Chinese-Text-Classification-Pytorch-master/THUCNews/saved_dict/TextCNN_64bot.ckpt"
    # checkpoint_path = "/data/gluo/Chinese-Text-Classification-Pytorch-master/THUCNews/saved_dict/TextCNN_64bot.ckpt"

    # 加载权重文件
    # checkpoint = torch.load(checkpoint_path)
    # print(checkpoint.keys())
    # 提取模型的 state_dict
    # model_state_dict = checkpoint

    # 将提取的 state_dict 加载到模型中
    # model.load_state_dict('model_state_dict')
    # model.to(device)
    test(config, model, test_iter)
