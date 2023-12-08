from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,roc_curve
from layer import RGTLayer
import pytorch_lightning as pl
from torch import nn
import torch
from Dataset import BotDataset
from torch.utils.data import DataLoader
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from os import listdir
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


def load_data(args):
    cat_features = torch.load(args.path + "det_new_cat_properties_tensor.pt", map_location="cpu")
    prop_features = torch.load(args.path + "det_new_num_properties_tensor.pt", map_location="cpu")
    tweet_features = torch.load(args.path + "det_new_tweets_tensor.pt", map_location="cpu")
    des_features = torch.load(args.path + "det_new_des_tensor.pt", map_location="cpu")
    
    tweet_features = tweet_features.view(14062, 100*768)
    x = torch.cat((cat_features, prop_features, tweet_features, des_features), dim=1)
    
    edge_index = torch.load(args.path + "det_edge_index.pt", map_location="cpu")
    edge_type = torch.load(args.path + "det_edge_type.pt", map_location="cpu").unsqueeze(-1)
    label = torch.load(args.path + "det_label.pt", map_location="cpu")
    data = Data(x=x, edge_index = edge_index, edge_attr=edge_type, y=label)
    
    datalen=label.shape[0]
    data.train_idx = torch.arange(0, int(datalen*0.7))
    data.valid_idx = torch.arange(int(datalen*0.7),int(datalen*0.9))
    data.test_idx = torch.arange(int(datalen*0.9), datalen)

    
    return data
    
class RGTDetector(pl.LightningModule):
    def __init__(self, args):
        super(RGTDetector, self).__init__()
    
        self.lr = args.lr
        self.l2_reg = args.l2_reg
    
        self.in_linear_numeric = nn.Linear(args.numeric_num, int(args.linear_channels/4), bias=True)
        self.in_linear_bool = nn.Linear(args.cat_num, int(args.linear_channels/4), bias=True)
        self.in_linear_tweet = nn.Linear(args.tweet_channel, int(args.linear_channels/4), bias=True)
        self.in_linear_des = nn.Linear(args.des_channel, int(args.linear_channels/4), bias=True)
        self.linear1 = nn.Linear(args.linear_channels, args.linear_channels)
        
        

        self.RGT_layer1 = RGTLayer(num_edge_type=2, in_channel=args.linear_channels, out_channel=args.out_channel, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)
        self.RGT_layer2 = RGTLayer(num_edge_type=2, in_channel=args.linear_channels, out_channel=args.out_channel, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)

        self.out1 = torch.nn.Linear(args.out_channel, 64)
        self.out2 = torch.nn.Linear(64, 2)

        self.drop = nn.Dropout(args.dropout)
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
        cat_features = train_batch.x[:, :args.cat_num]
        prop_features = train_batch.x[:, args.cat_num: args.cat_num + args.numeric_num]
        tweet_features = train_batch.x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.tweet_channel]
        des_features = train_batch.x[:, args.cat_num+args.numeric_num+args.tweet_channel: args.cat_num+args.numeric_num+args.tweet_channel+args.des_channel]
        label = train_batch.y
        
        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_attr.view(-1)
        
        user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
        user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
        user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
        user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
        
        user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
        user_features = self.drop(self.ReLU(self.linear1(user_features)))

        user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
        user_features = self.ReLU(self.RGT_layer2(user_features, edge_index, edge_type))

        user_features = self.drop(self.ReLU(self.out1(user_features)))
        pred = self.out2(user_features)
        loss = self.CELoss(pred, label.long())


        return loss
    
    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            cat_features = val_batch.x[:, :args.cat_num]
            prop_features = val_batch.x[:, args.cat_num: args.cat_num + args.numeric_num]
            tweet_features = val_batch.x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.tweet_channel]
            des_features = val_batch.x[:, args.cat_num+args.numeric_num+args.tweet_channel: args.cat_num+args.numeric_num+args.tweet_channel+args.des_channel]
            
            label = val_batch.y
        
            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_attr.view(-1)
            
            user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
            user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
            user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
            
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
            user_features = self.ReLU(self.RGT_layer2(user_features, edge_index, edge_type))

            user_features = self.drop(self.ReLU(self.out1(user_features)))
            pred = self.out2(user_features)
            pred_binary = torch.argmax(pred, dim=1)
            

            acc = accuracy_score(label.cpu(), pred_binary.cpu())
            f1 = f1_score(label.cpu(), pred_binary.cpu())
            
            self.log("val_acc", acc)
            self.log("val_f1", f1)

            print("acc: {} f1: {}".format(acc, f1))
    def forward(self, cat_features, prop_features, tweet_features,des_features, label,edge_index, edge_type):
        # 处理属性特征
        prop_features_linear = self.in_linear_numeric(prop_features)
        prop_features_relu = self.ReLU(prop_features_linear)
        user_features_numeric = self.drop(prop_features_relu)

        # 处理类别特征
        cat_features_linear = self.in_linear_bool(cat_features)
        cat_features_relu = self.ReLU(cat_features_linear)
        user_features_bool = self.drop(cat_features_relu)

        # 处理推文特征
        tweet_features_linear = self.in_linear_tweet(tweet_features)
        tweet_features_relu = self.ReLU(tweet_features_linear)
        user_features_tweet = self.drop(tweet_features_relu)

        # 处理描述特征
        des_features_linear = self.in_linear_des(des_features)
        des_features_relu = self.ReLU(des_features_linear)
        user_features_des = self.drop(des_features_relu)

        # 将不同的特征向量拼接在一起
        user_features_concat = torch.cat((user_features_numeric, user_features_bool, user_features_tweet, user_features_des), dim=1)

        # 进一步处理拼接后的特征向量
        user_features_linear1 = self.linear1(user_features_concat)
        user_features_relu1 = self.ReLU(user_features_linear1)
        user_features = self.drop(user_features_relu1)

        # 通过图神经网络层处理
        user_features_rgt1 = self.RGT_layer1(user_features, edge_index, edge_type)
        user_features_rgt1_relu = self.ReLU(user_features_rgt1)
        user_features_rgt2 = self.RGT_layer2(user_features_rgt1_relu, edge_index, edge_type)
        user_features_rgt2_relu = self.ReLU(user_features_rgt2)

        # 最终输出层
        user_features_out1 = self.out1(user_features_rgt2_relu)
        user_features_out1_relu = self.ReLU(user_features_out1)
        user_features_dropout = self.drop(user_features_out1_relu)
        pred = self.out2(user_features_dropout)
    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():

            cat_features = test_batch.x[:, :args.cat_num]
            prop_features = test_batch.x[:, args.cat_num: args.cat_num + args.numeric_num]
            tweet_features = test_batch.x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.tweet_channel]
            des_features = test_batch.x[:, args.cat_num+args.numeric_num+args.tweet_channel: args.cat_num+args.numeric_num+args.tweet_channel+args.des_channel]
            
            label = test_batch.y
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_attr.view(-1)
            
            # test: caculate flops
            import time
            start_time = time.time()
            # from fvcore.nn import FlopCountAnalysis
            # test_float = 0 
            # from thop import profile
            # macs, params = profile(self, inputs=(cat_features, prop_features,tweet_features,des_features,label,edge_index, edge_type) )
            # # flops = FlopCountAnalysis(self.forward, inputs=(cat_features, prop_features,tweet_features,des_features,label,edge_index, edge_type) )
            # print(macs, params)
            # exit(-1)
            user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
            user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
            user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
            
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
            user_features = self.ReLU(self.RGT_layer2(user_features, edge_index, edge_type))

            user_features = self.drop(self.ReLU(self.out1(user_features)))
            pred = self.out2(user_features)
            end_time = time.time()
            # print(f'times:{end_time-start_time}')
            
            pred_binary = torch.argmax(pred, dim=1)

            acc = accuracy_score(label.cpu(), pred_binary.cpu())
            f1 = f1_score(label.cpu(), pred_binary.cpu())
            precision =precision_score(label.cpu(), pred_binary.cpu())
            recall = recall_score(label.cpu(), pred_binary.cpu())
            auc = roc_auc_score(label.cpu(), pred[:,1].cpu())
            recall = recall_score(label.cpu(), pred_binary.cpu())

            self.log("acc", acc)
            self.log("f1",f1)
            self.log("precision", precision)
            self.log("recall", recall)
            self.log("auc", auc)

            print("acc: {} \t f1: {} \t precision: {} \t recall: {} \t auc: {}".format(acc, f1, precision, recall, auc))

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
# parser.add_argument("--path", type=str, default="./", help="dataset path")
parser.add_argument("--path", type=str, default="../../Datasets_process/Data/detptData/", help="dataset path")

parser.add_argument("--numeric_num", type=int, default=5, help="dataset path")
parser.add_argument("--linear_channels", type=int, default=128, help="linear channels")
parser.add_argument("--cat_num", type=int, default=3, help="catgorical features")
parser.add_argument("--des_channel", type=int, default=768, help="description channel")
parser.add_argument("--tweet_channel", type=int, default=768, help="tweet channel")
parser.add_argument("--out_channel", type=int, default=128, help="description channel")
parser.add_argument("--dropout", type=float, default=0.5, help="description channel")
parser.add_argument("--trans_head", type=int, default=8, help="description channel")
parser.add_argument("--semantic_head", type=int, default=8, help="description channel")
parser.add_argument("--batch_size", type=int, default=20, help="description channel")
parser.add_argument("--epochs", type=int, default=50, help="description channel")
parser.add_argument("--lr", type=float, default=1e-3, help="description channel")
parser.add_argument("--l2_reg", type=float, default=3e-5, help="description channel")
parser.add_argument("--random_seed", type=int, default=None, help="random")

if __name__ == "__main__":
    global args
    args = parser.parse_args()

    if args.random_seed != None:
        pl.seed_everything(args.random_seed)
        
    data = load_data(args)
    train_loader = NeighborLoader(data, num_neighbors=[4]*4, input_nodes=data.train_idx, batch_size=args.batch_size, shuffle=True)
    valid_loader = NeighborLoader(data, num_neighbors=[4]*4, input_nodes=data.valid_idx, batch_size=args.batch_size)# , num_workers=1
    test_loader = NeighborLoader(data, num_neighbors=[4]*4, input_nodes=data.test_idx, batch_size=args.batch_size)# , num_workers=1
    model = RGTDetector(args)
    trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=args.epochs, precision=16, log_every_n_steps=1)
    # train -------------------------------------
    # if you want to train the model 
    # trainer.fit(model, train_loader)

    # test --------------------------------------
    best_path = './lightning_logs/version_259/checkpoints/epoch=49-step=24650.ckpt'
    best_model = RGTDetector.load_from_checkpoint(checkpoint_path=best_path, args=args)
    trainer.test(best_model, test_loader, verbose=True)