from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,roc_curve
from tweet_layer import RGTLayer,Decoder
import pytorch_lightning as pl
from torch import nn
import torch
import json
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from os import listdir
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.nn import functional
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 




def load_data(args):
    args.path='../Datasets_process/Data/detptData/'
    with open(args.path+'vocab.json', 'r') as f:
        vocab=json.load(f)
        vocab_size=len(vocab)

    pre_sentences=[]
    new_sentences=[]
    with open('../Datasets_process/Data/detptData/det_behavior_data.txt', 'r',encoding='utf-8') as outfile:
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
    # print(len(sentences))
    print(args.path+"det_new_tweets_tensor.pt")
    tweets_tensor=torch.load(args.path+"det_new_tweets_tensor.pt",map_location="cpu")
    # print("tweets_tensor shape", tweets_tensor.shape)
    tweets_size,tweets_len,tweet_dim=tweets_tensor.shape
    end_tensor=torch.zeros(tweets_size,1,tweet_dim)
    # print(tweets_tensor.shape)
    tweets_tensor=torch.cat([tweets_tensor,end_tensor],1)
    tweets_tensor=torch.flatten(tweets_tensor,1,2)
    
    # print('tweets_tensor.shape',tweets_tensor.shape)
    for sen in sentences:
        temp=[vocab[n] for n in sen.split(' ')]
        pad_length=len(temp)
        dec_input=[[vocab['<S>']]+temp[:100]+[vocab['<SP>']]*(max_length-pad_length)]
        dec_output=[temp[:100]+[vocab['<SP>']]*(max_length-pad_length)+[vocab['<E>']]]
        
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
        
    dec_inputs=torch.LongTensor(dec_inputs)
    # dec_inputs = dec_inputs[:, :51]
    # print(dec_inputs)
    dec_outputs=torch.LongTensor(dec_outputs) 
    # print(dec)
    print("loading features...")
    cat_features = torch.load(args.path + "det_new_cat_properties_tensor.pt", map_location="cpu")
    # cat_features = torch.zeros(14062, 4,dtype=torch.float32)
    prop_features = torch.load(args.path + "det_new_num_properties_tensor.pt", map_location="cpu")
    
    # print(tweet_features.shape)
    # return
    des_features = torch.load(args.path + "det_new_des_tensor.pt", map_location="cpu")
    # dec_inputs = torch.load(args.path + "dec_inputs.pt", map_location="cpu")
    # dec_outputs = torch.load(args.path + "dec_outputs.pt", map_location="cpu")
    enc_inputs = torch.ones(14062,1).long()
    x = torch.cat((cat_features, prop_features, des_features,dec_inputs,enc_inputs,tweets_tensor), dim=1)
    # print('x.shape',x.shape)

    
    
  
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
    # print(args.path + "label.pt")
    # label=torch.load(args.path + "label.pt", map_location="cpu")
    # label=torch.cat((dec_outputs,tweet_labels),1)

    # print(label.shape)
    # return
    # label=torch.ones()
    # print(label)
    datalen=label.shape[0]
    # print(datalen)
    # print(label.shape)
    data = Data(x=x, edge_index = edge_index, edge_attr=edge_type, y=label)
    # print('data',data)

    print("loading index...")
    data.train_idx = torch.arange(0, int(datalen*0.7))
    data.valid_idx = torch.arange(int(datalen*0.7),int(datalen*0.9))
    data.test_idx = torch.arange(int(datalen*0.9), datalen)
    
    return data
    
class RGTDetector(pl.LightningModule):
    def __init__(self, args):
        super(RGTDetector, self).__init__()
    
        self.lr = args.lr
        self.l2_reg = args.l2_reg

        # self.in_linear_numeric = nn.Linear(args.numeric_num, int(args.linear_channels/4), bias=True)
        self.in_linear_numeric = nn.Linear(args.numeric_num, int(args.linear_channels/3), bias=True)

        # self.in_linear_bool = nn.Linear(args.cat_num, int(args.linear_channels/4), bias=True)
        self.in_linear_bool = nn.Linear(args.cat_num, int(args.linear_channels/3), bias=True)
        # self.in_linear_tweet = nn.Linear(args.tweet_channel, int(args.linear_channels/4), bias=True)
        # self.in_linear_des = nn.Linear(args.des_channel, int(args.linear_channels/4), bias=True)
        self.in_linear_des = nn.Linear(args.des_channel, int(args.linear_channels/3), bias=True)
        self.linear1 = nn.Linear(args.linear_channels, args.linear_channels)
        
        self.RGT_layer1 = RGTLayer(num_edge_type=2, in_channel=args.linear_channels, out_channel=args.out_channel, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)
        self.RGT_layer2 = RGTLayer(num_edge_type=2, in_channel=args.linear_channels, out_channel=args.out_channel, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)
        
        self.decoder = Decoder()
        
        self.out1 = torch.nn.Linear(args.out_channel, 64)
        self.out2 = torch.nn.Linear(64, 2)
        # self.projection = nn.Linear(args.linear_channels, 11, bias=False)
        self.projection = nn.Linear(args.linear_channels, 2, bias=False)
        
        # self.projection_tweet=nn.Linear(args.linear_channels, 701205, bias=False)
        

        self.drop = nn.Dropout(args.dropout)
        # self.CELoss = nn.CrossEntropyLoss()
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()
        
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def training_step(self, train_batch, batch_idx):
        # print()
        x,y=train_batch.x,train_batch.y
        # print(x.shape)
        cat_features = train_batch.x[:, :args.cat_num]
        # print('cat_features.shape',cat_features.shape)
        prop_features = train_batch.x[:, args.cat_num: args.cat_num + args.numeric_num] 
    
        des_features = train_batch.x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.des_channel]
        # print('des_features.shape',des_features.shape)
        dec_inputs = train_batch.x[:, args.cat_num+args.numeric_num+args.des_channel:args.cat_num+args.numeric_num+args.des_channel+args.dec_inputs_num]
        # print('dec_inputs.shape',dec_inputs.shape)
        enc_inputs = train_batch.x[:, args.cat_num+args.numeric_num+args.des_channel+args.dec_inputs_num : args.cat_num+args.numeric_num+args.des_channel+args.dec_inputs_num+args.enc_inputs_num ]
        # print('enc_inputs.shape',enc_inputs.shape)
        tweet_tensor = train_batch.x[:, args.cat_num+args.numeric_num+args.des_channel+args.dec_inputs_num+args.enc_inputs_num:args.cat_num+args.numeric_num+args.des_channel+args.dec_inputs_num+args.enc_inputs_num+args.tweet_embeddings_num]

        dec_inputs =dec_inputs.long()
        enc_inputs = enc_inputs.long()
        
        # 
        # 
        # 
        
        
        tweet_tensor_batch_size=tweet_tensor.shape[0]
   
        tweet_tensor=tweet_tensor.reshape(tweet_tensor_batch_size,101,768)

        label = train_batch.y
        label_behavior=label
        # label_behavior = label[:,:101]
        # label_tweet = label[:,101:]
        
        # print('label.shape,label_behaviors.shape',label.shape,label_behavior.shape,label_tweet.shape)
        
        edge_index = train_batch.edge_index
        # print(edge_index.shape)
        edge_type = train_batch.edge_attr.view(-1)
        
        user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
        
        # print('user_features_numeric.shape',user_features_numeric.shape,'prop_features.shape',prop_features.shape)
        user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
        # print('user_features_bool.shape',user_features_bool.shape,'cat_features.shape',cat_features.shape)
        # user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
        user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
        # print('user_features_des.shape',user_features_des.shape,'des_features.shape',des_features.shape)
        # print('user_features_des.shape',user_features_des.shape)
        # user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
        user_features = torch.cat((user_features_numeric,user_features_bool,user_features_des), dim = 1)
        # print('user_features.shape',user_features.shape)
        
        # print('user_features.shape',user_features.shape)
        user_features = self.drop(self.ReLU(self.linear1(user_features)))
        # print('user_features.shape',user_features.shape)

        user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
        # print('user_features.shape',user_features.shape)
        
        user_features = self.ReLU(self.RGT_layer2(user_features, edge_index, edge_type))
        # print('user_features.shape',user_features.shape)
        
        # print('user_features_2.shape',user_features.shape)
        user_features=user_features.unsqueeze(1)
        
        
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, user_features,tweet_tensor)
        # dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, user_features)
                                                                                                                         
        # print(dec_outputs.shape, dec_self_attns.shape, dec_enc_attns.shape)
        # print('dec_outputs.shape',dec_outputs.shape)
        
        dec_outputs=dec_outputs.mean(dim=1)
        # print('dec_outputs.shape',dec_outputs.shape)
        
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        # tweet_logits = self.projection_tweet(dec_outputs)
        # print(dec_logits.shape)
        # print('dec_logits.shape',dec_logits.shape)
        pred_behavior=dec_logits.view(-1, dec_logits.size(-1))
        # pred_tweet=tweet_logits.view(-1, tweet_logits.size(-1))
        
        # print('pred.shape',pred.shape)
        pred_behavior_binary = torch.argmax(pred_behavior, dim=1)
        # pred_tweet_binary = torch.argmax(pred_tweet, dim=1)
        
        # print('pred_behavior_binary.shape',pred_behavior_binary.shape)
        # print('pred_tweet_binary.shape',pred_tweet_binary.shape )
        # print('label.shape',label.shape)
        # print('label.view(-1).shape',label.view(-1).shape)
        
        # label_behaviors = label[:,:101]
        # label_tweets = label[:,101:]
        # print('label_behavior.reshape(-1).shape',label_behavior.reshape(-1).shape)
        # print('label_behavior.cpu().view(-1).shape',label_behavior.cpu().view(-1).shape)
        # print(label.view(-1).cpu(), pred_binary.cpu())
        # acc_behavior = accuracy_score(label_behavior.view(-1).cpu(), pred_behavior_binary.cpu())
        # acc_tweet = accuracy_score(label_tweet.view(-1).cpu(), pred_tweet_binary.cpu())
        acc_behavior = accuracy_score(label_behavior.reshape(-1).cpu(), pred_behavior_binary.cpu())
        # acc_tweet = accuracy_score(label_tweet.reshape(-1).cpu(), pred_tweet_binary.cpu())
        # print("train_acc", acc_behavior,acc_tweet)
        # print("train_acc", acc_behavior)
        
        
        f1_behavior = f1_score(label_behavior.reshape(-1).cpu(), pred_behavior_binary.cpu())
        # f1_tweet = f1_score(label_tweet.reshape(-1).cpu(), pred_tweet_binary.cpu(),average='macro')
            
        # print("train_f1", f1_behavior,f1_tweet)
        # print("train_f1", f1_behavior)
        
        # loss_behavior = self.CELoss(pred_behavior, label_behavior.reshape(-1))
        # print(pred_behavior, label_behavior.reshape(-1))
        # loss = self.CELoss(pred_behavior, label_behavior.reshape(-1))
        # print(dec_logits.shape, label_behavior.shape)
        loss = self.CELoss(dec_logits, label_behavior.long())
        
        # loss_tweet = self.CELoss(pred_tweet, label_tweet.reshape(-1))
        # loss=loss_behavior+loss_tweet
        # print(loss)
        # print(loss,loss_behavior,loss_tweet)
        

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            cat_features = val_batch.x[:, :args.cat_num]
            prop_features = val_batch.x[:, args.cat_num: args.cat_num + args.numeric_num]
            # tweet_features = val_batch.x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.tweet_channel]
            # des_features = val_batch.x[:, args.cat_num+args.numeric_num+args.tweet_channel: args.cat_num+args.numeric_num+args.tweet_channel+args.des_channel]
            des_features = val_batch.x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.des_channel]

            
            dec_inputs = val_batch.x[:, args.cat_num+args.numeric_num+args.des_channel:args.cat_num+args.numeric_num+args.des_channel+args.dec_inputs_num]
            enc_inputs = val_batch.x[:, args.cat_num+args.numeric_num+args.des_channel+args.dec_inputs_num:args.cat_num+args.numeric_num+args.des_channel+args.dec_inputs_num+args.enc_inputs_num ]
            tweet_tensor = val_batch.x[:, args.cat_num+args.numeric_num+args.des_channel+args.dec_inputs_num+args.enc_inputs_num:args.cat_num+args.numeric_num+args.des_channel+args.dec_inputs_num+args.enc_inputs_num+args.tweet_embeddings_num]
            
            dec_inputs =dec_inputs.long()
            enc_inputs = enc_inputs.long()
            
            tweet_tensor_batch_size=tweet_tensor.shape[0]
            tweet_tensor=tweet_tensor.reshape(tweet_tensor_batch_size,101,768)
            
            label = val_batch.y
            label_behavior = label[:,:101]
            label_tweet = label[:,101:]
            
            
            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_attr.view(-1)
            
            user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
            user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
            user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
            
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
            user_features = self.ReLU(self.RGT_layer2(user_features, edge_index, edge_type))
            
            user_features=user_features.unsqueeze(1)
            # dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, user_features)
            dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, user_features,tweet_tensor)
            
                                                                                                                            
            # print(dec_outputs.shape, dec_self_attns.shape, dec_enc_attns.shape)
            # dec_logits = self.projection(label) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
            # print('dec_logits.shape',dec_logits.shape)
            
            dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
            # tweet_logits = self.projection_tweet(dec_outputs)
            
            # print('dec_logits.shape',dec_logits.shape)
            pred_behavior=dec_logits.view(-1, dec_logits.size(-1))
            # pred_tweet=tweet_logits.view(-1, tweet_logits.size(-1))
            
            # print('pred.shape',pred.shape)
            pred_behavior_binary = torch.argmax(pred_behavior, dim=1)
            # pred_tweet_binary = torch.argmax(pred_tweet, dim=1)
            
            # pred=dec_logits.view(-1, dec_logits.size(-1))

            # user_features = self.drop(self.ReLU(self.out1(user_features)))
            # pred = self.out2(user_features)
            # print(pred.size())
            # pred_binary = torch.argmax(pred, dim=1)
            
            # print(self.label[val_batch].size())

            acc_behavior = accuracy_score(label_behavior.reshape(-1).cpu(), pred_behavior_binary.cpu())
            # acc_tweet = accuracy_score(label_tweet.reshape(-1).cpu(), pred_tweet_binary.cpu())
            # print("val_acc", acc_behavior,acc_tweet)
            print("val_acc", acc_behavior)
            
            f1_behavior = f1_score(label_behavior.reshape(-1).cpu(), pred_behavior_binary.cpu())
            # f1_tweet = f1_score(label_tweet.reshape(-1).cpu(), pred_tweet_binary.cpu(),average='macro')
                
            # print("val_f1", f1_behavior,f1_tweet)
            print("val_f1", f1_behavior)
            
            # loss_behavior = self.CELoss(pred_behavior, label_behavior.reshape(-1))
            loss = self.CELoss(pred_behavior, label_behavior.reshape(-1))
            # loss_tweet = self.CELoss(pred_tweet, label_tweet.reshape(-1))
            # loss=loss_behavior+loss_tweet
            print(loss)
            # print(loss,loss_behavior,loss_tweet)

            # acc = accuracy_score(label.view(-1).cpu(), pred_binary.cpu())
        
            # print("val_acc", acc)
            
            # f1 = f1_score(label.view(-1).cpu(), pred_binary.cpu(),average='macro')
                
            # print("val_f1", f1)
            
            # self.log("val_acc", (acc_behavior,acc_tweet), prog_bar=True)
            # self.log("val_f1", (f1_behavior,f1_tweet), prog_bar=True)
            
            self.log("val_acc", acc_behavior, prog_bar=True)          
            self.log("val_f1", f1_behavior, prog_bar=True)

            print("acc: {} f1: {}".format(acc, f1))
    
    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            cat_features = test_batch.x[:, :args.cat_num]
            prop_features = test_batch.x[:, args.cat_num: args.cat_num + args.numeric_num]
            # tweet_features = test_batch.x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.tweet_channel]
            # des_features = test_batch.x[:, args.cat_num+args.numeric_num+args.tweet_channel: args.cat_num+args.numeric_num+args.tweet_channel+args.des_channel]
            des_features = test_batch.x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.des_channel]
            
            dec_inputs = test_batch.x[:, args.cat_num+args.numeric_num+args.des_channel:args.cat_num+args.numeric_num+args.des_channel+args.dec_inputs_num]
            enc_inputs = test_batch.x[:, args.cat_num+args.numeric_num+args.des_channel+args.dec_inputs_num:args.cat_num+args.numeric_num+args.des_channel+args.dec_inputs_num+args.enc_inputs_num ]
            tweet_tensor = test_batch.x[:, args.cat_num+args.numeric_num+args.des_channel+args.dec_inputs_num+args.enc_inputs_num:args.cat_num+args.numeric_num+args.des_channel+args.dec_inputs_num+args.enc_inputs_num+args.tweet_embeddings_num]

            tweet_tensor_batch_size=tweet_tensor.shape[0]
            tweet_tensor=tweet_tensor.reshape(tweet_tensor_batch_size,101,768)
            
            dec_inputs =dec_inputs.long()
            enc_inputs = enc_inputs.long()
            
            
            # label = test_batch.y[:args.test_batch_size]
            # print('label.shape')
            # label= test_batch.y
            # label_behavior = label[:,:101]
            # label_tweet = label[:,101:]
            label = test_batch.y
            label_behavior=label
            
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_attr.view(-1)
            # print('edge_index.shape,edge_type.shape',edge_index.shape,edge_type.shape)
            # print('prop_features.shape',prop_features.shape)
            
            user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
            user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
            # user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
            user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
            
            # user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
            user_features = self.ReLU(self.RGT_layer2(user_features, edge_index, edge_type))

            user_features=user_features.unsqueeze(1)
            print('user_features.shape',user_features.shape)
            # dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, user_features)
            dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, user_features,tweet_tensor)
            dec_outputs=dec_outputs.mean(dim=1)
            
            
            dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
            # tweet_logits = self.projection_tweet(dec_outputs)
            
            # print('dec_logits.shape',dec_logits.shape)
            pred_behavior=dec_logits.view(-1, dec_logits.size(-1))
            # pred_tweet=tweet_logits.view(-1, tweet_logits.size(-1))
            
            # print('pred.shape',pred.shape)
            pred_behavior_binary = torch.argmax(pred_behavior, dim=1)
            # pred_tweet_binary = torch.argmax(pred_tweet, dim=1)
            
            # print(dec_outputs.shape, dec_self_attns.shape, dec_enc_attns.shape)
            # dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
            # print('dec_logits.shape',dec_logits.shape)
            # pred=dec_logits.view(-1, dec_logits.size(-1))
            # print('dec_logits.shape',dec_logits.shape)

            # print('pred.shape', pred.shape)

            # user_features = self.drop(self.ReLU(self.out1(user_features)))
            # pred = self.out2(user_features)[:args.test_batch_size]
            # pred_binary = torch.argmax(pred, dim=1)
            # print('pred_binary.shape',pred_binary.shape )
            # print('label.shape',label.shape)
            # print('pred_binary.shape', pred_binary.shape,'label.view(-1).shape',label.view(-1).shape)
            
            pred_behavior_test.append(pred_behavior_binary.squeeze().cpu())
            pred_behavior_test_prob.append(pred_behavior[:,1].squeeze().cpu())
            label_behavior_test.append(label_behavior.squeeze().cpu())
            
            # pred_tweet_test.append(pred_tweet_binary.squeeze().cpu())
            # pred_tweet_test_prob.append(pred_tweet[:,1].squeeze().cpu())
            # label_tweet_test.append(label_tweet.squeeze().cpu())
            
            
            # acc = accuracy_score(label.view(-1).cpu(), pred_binary.cpu())
            acc_behavior = accuracy_score(label_behavior.reshape(-1).cpu(), pred_behavior_binary.cpu())
            # acc_tweet = accuracy_score(label_tweet.reshape(-1).cpu(), pred_tweet_binary.cpu())
        
            # print("test_acc", acc_behavior,acc_tweet)
            print("test_acc", acc_behavior)
            
            # f1_behavior = f1_score(label_behavior.reshape(-1).cpu(), pred_behavior_binary.cpu(),average='macro')
            f1_behavior = f1_score(label_behavior.reshape(-1).cpu(), pred_behavior_binary.cpu())
            
            # f1_tweet = f1_score(label_tweet.reshape(-1).cpu(), pred_tweet_binary.cpu(),average='macro')
                     
            # f1 = f1_score(label.view(-1).cpu(), pred_binary.cpu(),average='macro')
                
            # print("test_f1", f1_behavior,f1_tweet)
            print("test_f1", f1_behavior)

            # acc = accuracy_score(label.cpu(), pred_binary.cpu())
            # f1 = f1_score(label.cpu(), pred_binary.cpu())
            precision_behavior =precision_score(label_behavior.reshape(-1).cpu(), pred_behavior_binary.cpu())
            # precision_tweet =precision_score(label_tweet.reshape(-1).cpu(), pred_tweet_binary.cpu(),average='macro')
            
            print('precision',precision_behavior)
            # print('precision',precision_behavior,precision_tweet)
            
            recall_behavior = recall_score(label_behavior.reshape(-1).cpu(), pred_behavior_binary.cpu())
            # recall_tweet = recall_score(label_tweet.reshape(-1).cpu(), pred_tweet_binary.cpu(),average='macro')
            
            # print('recall',recall_behavior,recall_tweet)
            print('recall',recall_behavior)
            # auc = roc_auc_score(label.view(-1).cpu(), pred[:,1].cpu(),average='macro')
            auc = roc_auc_score(label_behavior.reshape(-1).cpu(), pred_behavior_binary.cpu())

            self.log("acc_behavior", acc_behavior)
            self.log("f1_behavior",f1_behavior)
            self.log("precision_behavior",precision_behavior)
            self.log("recall_behavior", recall_behavior)
            self.log("auc", auc)

            # print("acc: {} \t f1: {} \t precision: {} \t recall: {} \t auc: {}".format(acc, f1, precision, recall, auc))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }


parser = argparse.ArgumentParser(description="Reproduction of Heterogeneity-aware Bot detection with Relational Graph Transformers")
parser.add_argument("--path", type=str, default="/data/gluo/Bots_code/RGCN_prediction/Data/detptData/", help="dataset path")
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
parser.add_argument("--batch_size", type=int, default=30, help="description channel")
parser.add_argument("--epochs", type=int, default=10, help="description channel")
parser.add_argument("--lr", type=float, default=5e-4, help="description channel")
parser.add_argument("--l2_reg", type=float, default=1e-5, help="description channel")
parser.add_argument("--random_seed", type=int, default=None, help="random")
parser.add_argument("--test_batch_size", type=int, default=128, help="random")

if __name__ == "__main__":
    global args, pred_test, pred_test_prob, label_test
    args = parser.parse_args()
    pred_behavior_test = []
    pred_behavior_test_prob = []
    label_behavior_test = []
    
       
    if args.random_seed != None:
        pl.seed_everything(args.random_seed)
        
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename='{val_acc:.4f}',
        save_top_k=1,
        verbose=True)

    # load dataset
    data = load_data(args)
    print("loading...")
    print('Train epochs',args.epochs)
    # num_neighbors=[256]*4
    # train_loader = NeighborLoader(data, num_neighbors=[4]*4, input_nodes=data.train_idx, batch_size=args.batch_size, shuffle=True)
    
    # data_loader
    train_loader = NeighborLoader(data, num_neighbors=[4]*4, input_nodes=data.train_idx, batch_size=args.batch_size, shuffle=True)
    valid_loader = NeighborLoader(data, num_neighbors=[4]*4, input_nodes=data.valid_idx, batch_size=args.batch_size)# , num_workers=1
    test_loader = NeighborLoader(data, num_neighbors=[4]*4, input_nodes=data.test_idx, batch_size=args.test_batch_size)# , num_workers=1
    
    # model
    model = RGTDetector(args)


    # Train ----------------------------------------------------------------
    trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=args.epochs, precision=16, log_every_n_steps=1,check_val_every_n_epoch=1)#, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader)#, valid_loader)

    # Test -----------------------------------------------------------------
    dir = './lightning_logs/version_{}/checkpoints/'.format(trainer.logger.version)
    best_path = './lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[0])
    best_model = RGTDetector.load_from_checkpoint(checkpoint_path=best_path, args=args)
    trainer.test(best_model, test_loader, verbose=True)
    