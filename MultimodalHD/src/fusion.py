import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import random
import copy






class attention_module(nn.Module):
    def __init__(self,configs,keep_idx):
        super().__init__()
        self.configs = configs
        self.keep_idx = keep_idx
        self.input_dim = int(configs['config']['D'])
        self.dataset = configs['config']['dataset']
        self.embedding_dim = int(configs['fusion']['embedding_dim'])
        self.n_heads = int(configs['fusion']['n_heads'])
        self.head_dim = self.embedding_dim // self.n_heads
        assert self.head_dim * self.n_heads == self.embedding_dim
        if self.dataset == "HAR":
            self.num_readings=9
            self.num_class = 6
        if self.dataset == "MHEALTH":
            self.num_readings=21
            self.num_class = 13
        if self.dataset == "OPP":
            self.num_readings=39
            self.num_class = 17

        self.projection = nn.Linear(self.input_dim,self.embedding_dim,bias = True)

        self.classification_token = nn.Parameter(torch.rand(self.embedding_dim),requires_grad = True)
        #self.positional_encoding = nn.Parameter(self.get_positional_encoding(),requires_grad = False)
        self.positional_encoding = nn.Parameter(torch.rand(self.num_readings+1,self.embedding_dim),requires_grad = True)
        self.layer_norm1 = nn.LayerNorm([self.embedding_dim], elementwise_affine=False)
        self.key_weights = nn.Linear(self.embedding_dim,self.embedding_dim,bias= False)
        self.query_weights = nn.Linear(self.embedding_dim,self.embedding_dim,bias= False)
        self.value_weights = nn.Linear(self.embedding_dim,self.embedding_dim,bias= False)
        self.layer_norm2 = nn.LayerNorm([self.embedding_dim], elementwise_affine=False)
        self.classifier_fc1 = nn.Linear(self.embedding_dim, 25)
        self.classifier_fc2 = nn.Linear(25, self.num_class)

        self.dropout = nn.Dropout(0.25)

    def get_positional_encoding(self):
        position = torch.arange(self.num_readings+1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2) * (-math.log(10000.0) / self.embedding_dim))
        positional_encoding = torch.zeros(self.num_readings+1, 1, self.embedding_dim)
        positional_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.squeeze()
        return positional_encoding



    def forward(self,x):
        batch_size = x.shape[0]
        x = x[:,self.keep_idx,:]

        x = self.projection(x)

        x = torch.tanh(x)

        x = torch.cat((self.classification_token.repeat(batch_size,1,1),x),1)

        pos_enc_idx = [0] + [i+1 for i in self.keep_idx]
        x = x + self.positional_encoding[pos_enc_idx,:].repeat(batch_size,1,1)

        assert x.shape[1] == len(self.keep_idx)+1

        x = self.layer_norm1(x)

        key = self.key_weights(x)
        query = self.query_weights(x)
        value = self.value_weights(x)

        keys = key.reshape(batch_size, len(self.keep_idx)+1, self.n_heads, self.head_dim)
        queries = query.reshape(batch_size, len(self.keep_idx)+1, self.n_heads, self.head_dim)
        values = value.reshape(batch_size, len(self.keep_idx)+1, self.n_heads, self.head_dim)

        attns = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attns = torch.softmax(attns / (self.head_dim ** (1 / 2)), dim=3)

        attn_out = torch.einsum("nhql,nlhd->nqhd", [attns, values]).reshape(
                batch_size, len(self.keep_idx)+1, self.n_heads * self.head_dim)
        classification_tokens = (attn_out / (len(self.keep_idx)+1))[:,0,:]

        classification_tokens = self.layer_norm2(classification_tokens)

        out = torch.tanh(self.classifier_fc1(classification_tokens))

        out = self.dropout(out)

        out = self.classifier_fc2(out)

        return out


