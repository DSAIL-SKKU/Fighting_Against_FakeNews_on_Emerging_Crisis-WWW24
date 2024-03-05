import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pandas as pd
import numpy as np
from pprint import pprint
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from transformers import XLNetTokenizer, XLNetModel, XLNetForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from collections import Counter
import random
import argparse

# dgl library
import sys
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn import SAGEConv
from pathlib import Path
from collections import Counter

from scipy.sparse import coo_matrix

# torch:
from torch.utils.data import Dataset, DataLoader, TensorDataset,random_split
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report

# clean text:
import re
import emoji
from soynlp.normalizer import repeat_normalize
from soynlp.normalizer import *
from collections import Counter
emojis = ''.join(emoji.UNICODE_EMOJI.keys())
pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
username_pattern = re.compile(
    r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)")
url_pattern = re.compile(
    r"[-a-zA-Z0-9@:%_\+.~#?&//=]{2,256}\.[a-z]{2,4}\b(\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?")

def clean(x):
        x = pattern.sub(' ', x)
        x = username_pattern.sub('', x)
        x = url_pattern.sub('', x)
        x = repeat_normalize(x, num_repeats=2)
        x = x.strip()
        return x

# Parameters
class Arg:
    random_seed: int = 20201203  # Random Seed
    pretrained_model =  'xlnet-base-cased' 
    pretrained_tokenizer = 'xlnet-base-cased' 
    cache_dir = './cache' 
    

    train_data_path = './data/train_ex2.pkl'
    test_data_path = './data/test.pkl'

    train_edge_path = './data/train_ex2.npy'
    test_edge_path = './data/test.npy'

    
    log_dir = './report/'
    true_pred_dir = './'
    log_name = 'xln-uncased-fakenews-text-feature'
    version = 2
    batch_size: int = 32  # Optional, Train/Eval Batch Size. Overrides `auto_batch_size` 
    lr: float = 2e-5  # Starting Learning Rate
    epochs: int = 10  # Max Epochs
    max_length: int = 120  # Max Length input size
    report_cycle: int = 30  # Report (Train Metrics) Cycle
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    optimizer: str = 'AdamW'  # AdamW vs AdamP
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    hidden_dropout_prob = 0.1
    hidden_size = 768 

    # graphSAGE args
    graph_dropout_prob = 0.6
    graph_in_channels : int = 1536 # 768*2
    graph_out_channels : int = 8 # 8
    num_class : int = 2 # label
    negative_slope : float = 0.2
    num_heads : int = 2
    aggre_list = ['lstm','gcn']
    drop_rate = [0.0,0.1,0.2,0.3]
    p = 0.5


# Prepare for Graph Embedding
class MyGraphDataset(Dataset):
    def __init__(self, xlnet_output, adj_mat, batch_size, batch_index):        
        self.features = xlnet_output
        
        batch_start = batch_index * batch_size
        batch_mat = adj_mat[batch_start:batch_start+batch_size, batch_start:batch_start+batch_size]
        
        self.batch_adj_edge = np.nonzero(adj_mat[batch_start:batch_start+batch_size, batch_start:batch_start+batch_size])
        self.g = self.dgl_homog(self.batch_adj_edge[0],self.batch_adj_edge[1], batch_mat.shape[0])        

    def __len__(self):
        return len(xlnet_output.size(0))
    
    def dgl_homog(self, edge_list1, edge_list2, batch_size):
        # homogeneous graph
        g = dgl.graph((edge_list1,edge_list2),num_nodes=batch_size)
        g = dgl.add_self_loop(g)
        return g

    def __getitem__(self, idx):  # we just fetch a single graph
        return self.g

    
class Model(LightningModule):
    def __init__(self, options, random_seed):
        super().__init__()
        # config:
        self.args = options
        self.random_seed = random_seed
        self.batch_size = self.args.batch_size
        # meta data:
        self.epochs_index = 0
        self.label_cols = 'label'
        self.num_labels = 2
        self.best_f1 = 0
        # for gcn
        self.aggre_type = self.args.aggre_list[0] # [0]: lstm, [1]: gcn
        self.drop_rate = self.args.drop_rate[0] # drop_rate = [0.0,0.1,0.2,0.3]
        
        # modules:
        # XLNet
        self.tokenizer = XLNetTokenizer.from_pretrained(self.args.pretrained_tokenizer, cache_dir=self.args.cache_dir)
        self.xlnet_data = XLNetModel.from_pretrained(self.args.pretrained_model, cache_dir=self.args.cache_dir)
        self.xlnet_utu = XLNetModel.from_pretrained(self.args.pretrained_model, cache_dir=self.args.cache_dir)

        # MultiheadAttention
        self.utu_mhatt = nn.MultiheadAttention(self.args.hidden_size, self.args.num_heads)

        self.utu_ffn = nn.Sequential(
            nn.Linear(self.args.hidden_size, self.args.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.args.hidden_dropout_prob),
        )
        self.utu_preclf = nn.Sequential( 
            nn.Linear(self.args.hidden_size * 2, self.args.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.args.hidden_dropout_prob)
        )

        self.in_channels = self.args.hidden_size*2+100
        
        self.fc1 = nn.Linear(self.args.hidden_size*3, self.args.hidden_size)
        self.layer_norm = nn.ModuleList([nn.LayerNorm(self.args.hidden_size) for _ in range(2)])
    
        # GraphSAGE
        self.conv1 = SAGEConv(self.in_channels, int(self.in_channels/2), aggregator_type = self.aggre_type, feat_drop = self.drop_rate)
        self.conv2 = SAGEConv(int(self.in_channels/2), int(self.in_channels/4), aggregator_type = self.aggre_type, feat_drop = self.drop_rate)
        self.conv3 = SAGEConv(int(self.in_channels/4), int(self.in_channels/8), aggregator_type = self.aggre_type, feat_drop = self.drop_rate)
        self.conv4 = SAGEConv(int(self.in_channels/8), int(self.in_channels/16), aggregator_type = self.aggre_type, feat_drop = self.drop_rate)
        
        self.lin = torch.nn.Linear(int(self.in_channels/16), self.num_labels)
    
        self.dropout = nn.Dropout(self.args.hidden_dropout_prob)
        self.classifier = nn.Linear(self.args.hidden_size, self.num_labels)
        
        print('model initialized')


    def forward(self,data, entity, utube0, utube1, batch_index, **kwargs):
        # News Claim (Title) - XLNet (Content Embedding)
        outputs_data = self.xlnet_data(input_ids=data, **kwargs)
        outputs_data = outputs_data[0][:,-1]

        # YouTube Text - XLNet (Propagated Information Encoder)
        output_utube0 = self.xlnet_utu(input_ids=utube0, **kwargs)
        output_utube1 = self.xlnet_utu(input_ids=utube1, **kwargs)
        
        output_utube0 = output_utube0[0]
        output_utube1 = output_utube1[0]
        
        h = torch.cat((output_utube0, output_utube1), axis=1) 

        # YouTube Text - Attention (Propagated Information Encoder)
        P, _= self.utu_mhatt(query=h, key=h, value=h)

        ytb_output = self.layer_norm[0](h + P) # residual->layer normalization
        ytb_output = self.layer_norm[1](ytb_output + self.utu_ffn(ytb_output)) # feed forward + residual ->layer normalization

        # bring each <CLS> token and concat
        ytb_meta = torch.cat((ytb_output[:,self.args.max_length-1], ytb_output[:,-1]), 1)
        ytb_self = self.utu_preclf(ytb_meta)
        output = torch.cat([outputs_data, entity, ytb_self], dim=1)
        

        # Graph Embedding
        g = MyGraphDataset(output, self.adj_mat, self.batch_size, batch_index)[0].to('cuda:1')

        x = self.conv1(g, output) # g, feat
        x = self.conv2(g, x) # g, feat
        x = self.conv3(g, x) # g, feat
        x = self.conv4(g, x) # g, feat

        x = F.dropout(x, p=self.args.p, training=self.training)
        logits = self.lin(x)

        return logits


    def configure_optimizers(self):
        if self.args.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'AdamP':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')
        if self.args.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.args.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
    
    def preprocess_dataframe(self, df,col_name):
        df[col_name] = df[col_name].map(lambda x: clean(str(x)))
        return df
    
    
    def prepare_data(self):
        train_edge_index = np.load(self.args.train_edge_path)
        test_edge_index = np.load(self.args.test_edge_path)

        train_df = pd.read_pickle(self.args.train_data_path)
        test_df = pd.read_pickle(self.args.test_data_path)
        print(len(train_df),len(test_df))
        
        self.train_adj_mat = np.zeros((len(train_df), len(train_df)))
        for i in range(len(train_edge_index[0])):
            self.train_adj_mat[train_edge_index[0][i], train_edge_index[1][i]] = 1
        self.test_adj_mat = np.zeros((len(test_df),len(test_df)))
        for i in range(len(test_edge_index[0])):
            self.test_adj_mat[test_edge_index[0][i], test_edge_index[1][i]] = 1
        

    def train_dataloader(self):
        self.adj_mat = self.train_adj_mat

        with open(self.args.train_data_path, 'rb') as f:
            df = pickle.load(f)

        df = self.preprocess_dataframe(df, 'title')
        df = self.preprocess_dataframe(df, 'youtube0')
        df = self.preprocess_dataframe(df, 'youtube1')
        
        
        df['XLNet_text'] = df['title'].map(lambda x: self.tokenizer.encode(
            str(x),
            padding = 'max_length',
            max_length=self.args.max_length,
            truncation=True,
        ))
        
        df['utu_0'] = df['youtube0'].map(lambda x: self.tokenizer.encode(
            str(x),
            padding = 'max_length',
            max_length=self.args.max_length,
            truncation=True,
        ))
        
        df['utu_1'] = df['youtube1'].map(lambda x: self.tokenizer.encode(
            str(x),
            padding = 'max_length',
            max_length=self.args.max_length,
            truncation=True,
        ))
        
        pprint(f"Train Size: {len(df)}")
        

        dataset = TensorDataset(
            torch.tensor(df['XLNet_text'].tolist(), dtype=torch.long),
            torch.tensor(df['ner_onehot'].tolist(), dtype=torch.long),
            torch.tensor(df['utu_0'].tolist(), dtype=torch.long),
            torch.tensor(df['utu_1'].tolist(), dtype=torch.long),
            
            torch.tensor(df[self.label_cols].to_list(), dtype=torch.long),
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )


    def test_dataloader(self):
        self.adj_mat = self.test_adj_mat

        with open(self.args.test_data_path, 'rb') as f:
            df = pickle.load(f)        
        
        df = self.preprocess_dataframe(df, 'title')
        df = self.preprocess_dataframe(df, 'youtube0')
        df = self.preprocess_dataframe(df, 'youtube1')
        
        
        df['XLNet_text'] = df['title'].map(lambda x: self.tokenizer.encode(
            str(x),
            padding = 'max_length',
            max_length=self.args.max_length,
            truncation=True,
        ))
        
        df['utu_0'] = df['youtube0'].map(lambda x: self.tokenizer.encode(
            str(x),
            padding = 'max_length',
            max_length=self.args.max_length,
            truncation=True,
        ))
        
        df['utu_1'] = df['youtube1'].map(lambda x: self.tokenizer.encode(
            str(x),
            padding = 'max_length',
            max_length=self.args.max_length,
            truncation=True,
        ))

        pprint(f"Test Size: {len(df)}")

        dataset = TensorDataset(
            torch.tensor(df['XLNet_text'].tolist(), dtype=torch.long),
            torch.tensor(df['ner_onehot'].tolist(), dtype=torch.long),
            
            torch.tensor(df['utu_0'].tolist(), dtype=torch.long),
            torch.tensor(df['utu_1'].tolist(), dtype=torch.long),
            
            torch.tensor(df[self.label_cols].to_list(), dtype=torch.long),
        )
        

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )
        

    def training_step(self, batch, batch_idx):
        self.adj_mat = self.train_adj_mat
        
        data, entity, utube0, utube1, labels = batch  
        logits = self(data, entity, utube0, utube1, batch_idx)    

        loss = None
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))    

        return {'loss': loss}


    def test_step(self, batch, batch_idx):
        self.adj_mat = self.test_adj_mat

        data, entity, utube0, utube1, labels = batch  
        logits = self(data, entity, utube0, utube1, batch_idx) 

        loss = None
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())


        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }


    def test_epoch_end(self, outputs):
        loss = torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        _loss = loss / len(outputs)

        loss = float(_loss)
        y_true = []
        y_pred = []
        

        for i in outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']
            
        # save - classification_report
        result = classification_report(y_true, y_pred,digits=4,output_dict=True)
        df = pd.DataFrame(result).transpose()
        df.to_csv(f'./result_ex2.csv')
            
            
        y_pred = np.asanyarray(y_pred)
        y_true = np.asanyarray(y_true)
        
          
        # save - y_pred, y_true
        if self.epochs_index == self.args.epochs -1:
            pickle.dump(y_pred, open(os.path.join(self.args.log_dir, self.args.log_name, f'version_{self.logger.version}', 
                                    f'ex1_pred_{self.label_cols}_epoch_{self.epochs_index}.p'), 'wb'))
            pickle.dump(y_true, open(os.path.join(self.args.log_dir, self.args.log_name, f'version_{self.logger.version}', 
                                    f'ex1_true_{self.label_cols}_epoch_{self.epochs_index}.p'), 'wb'))

        

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    args = parser.parse_args()
    return args

def main(opt):
    
    args = Arg()
    print("Using PyTorch Ver", torch.__version__)
    seed_everything(args.random_seed)
    
    model = Model(args, args.random_seed)
    logger = TensorBoardLogger(
    save_dir=args.log_dir,
    version=args.version,
    name=args.log_name
    )
    print(":: Start Training ::")
    trainer = Trainer(
        logger = logger,
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        deterministic=True,
        # For GPU Setup
        gpus=[1] if torch.cuda.is_available() else None, 
        precision=16 if args.fp16 else 32
    )
    trainer.fit(model)
    trainer.test(model, model.test_dataloader())

if __name__ == '__main__':
    opt = get_args()
    
    main(opt)