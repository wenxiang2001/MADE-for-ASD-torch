#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn 
import torch.nn.functional as F


class AutoEncoder(torch.nn.Module):
    """
    Autoencoder model: input_size -> code_size -> input_size
    Supports tight weights and corruption.
    """
    def __init__(self, input_size, code_size, corruption=0.0, tight=False,
                 enc_activation=torch.tanh, dec_activation=torch.tanh):
        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        self.code_size = code_size
        self.corruption = corruption
        self.tight = tight
        self.enc_activation = enc_activation
        self.dec_activation = dec_activation
        
        # 编码器
        self.W_enc = torch.nn.Parameter(torch.Tensor(input_size, code_size))
        self.b_enc = torch.nn.Parameter(torch.zeros(code_size))
        
        # 如果不是紧密权重，创建解码器权重参数
        if not tight:
            self.W_dec = torch.nn.Parameter(torch.Tensor(code_size, input_size))
        
        self.b_dec = torch.nn.Parameter(torch.zeros(input_size))
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        # 使用与TF相同的初始化参数和随机种子
        torch.manual_seed(42)  # 设置与TF相同的种子
        torch.nn.init.uniform_(self.W_enc, 
                        -6.0 / math.sqrt(self.input_size + self.code_size),
                        6.0 / math.sqrt(self.input_size + self.code_size))
        
        if not self.tight:
            torch.nn.init.uniform_(self.W_dec, 
                            -6.0 / math.sqrt(self.code_size + self.input_size),
                            6.0 / math.sqrt(self.code_size + self.input_size))
    
    def encode(self, x):
        if self.corruption > 0.0:
            # 改为使用均匀分布，更接近TF实现
            noise_mask = torch.rand_like(x)  # 生成[0,1)之间的随机数
            # 只保留小于(1-corruption)的值，实现与TF相同的效果
            x = x * (noise_mask < (1 - self.corruption)).float()
        
        encode = torch.matmul(x, self.W_enc) + self.b_enc
        if self.enc_activation is not None:
            encode = self.enc_activation(encode)
        return encode
    
    def decode(self, encode):
        if self.tight:
            # 使用编码器权重的转置
            W_dec = self.W_enc.t()
        else:
            W_dec = self.W_dec
        
        decode = torch.matmul(encode, W_dec) + self.b_dec
        if self.dec_activation is not None:
            decode = self.enc_activation(decode)
        return decode
    
    def forward(self, x):
        encode = self.encode(x)
        decode = self.decode(encode)
        return encode, decode
    
    def get_cost(self, x, decode):
        return torch.sqrt(torch.mean(torch.square(x - decode)))
    
    def get_params(self):
        params = {
            "W_enc": self.W_enc,
            "b_enc": self.b_enc,
            "b_dec": self.b_dec,
        }
        if not self.tight:
            params["W_dec"] = self.W_dec
        return params


class MultiLayerNN(torch.nn.Module):
    """
    Multi-layer model
    Supports pre-training initialization.
    """
    def __init__(self, input_size, n_classes, layers, init=None):
        super(MultiLayerNN, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.layers_config = layers
        
        # 创建层
        self.layer_modules = torch.nn.ModuleList()
        self.dropouts = []
        self.params = {}
        
        current_size = input_size
        
        for i, layer in enumerate(layers):
            layer_size = layer["size"]
            
            # 创建线性层
            if init is None:
                linear = torch.nn.Linear(current_size, layer_size)
                torch.nn.init.zeros_(linear.weight)
                torch.nn.init.zeros_(linear.bias)
            else:
                linear = torch.nn.Linear(current_size, layer_size)
                linear.weight.data = torch.tensor(init[i]["W"], dtype=torch.float32).T
                linear.bias.data = torch.tensor(init[i]["b"], dtype=torch.float32)
            
            self.layer_modules.append(linear)
            
            # 激活函数
            activation = None
            if "actv" in layer and layer["actv"] is not None:
                if layer["actv"] == torch.tanh:
                    activation = torch.nn.Tanh()
                elif layer["actv"] == torch.sigmoid:
                    activation = torch.nn.Sigmoid()
                elif layer["actv"] == F.relu:
                    activation = torch.nn.ReLU()
                
            if activation is not None:
                self.layer_modules.append(activation)
            
            # 添加dropout
            dropout_rate = 0.5  # 默认dropout率，可以通过forward传入具体的dropout率
            dropout = torch.nn.Dropout(dropout_rate)
            self.layer_modules.append(dropout)
            self.dropouts.append(dropout)
            
            # 更新当前大小
            current_size = layer_size
            
            # 存储参数
            self.params["W_" + str(i+1)] = linear.weight
            self.params["b_" + str(i+1)] = linear.bias
        
        # 输出层
        self.output_layer = torch.nn.Linear(current_size, n_classes)
        torch.nn.init.uniform_(self.output_layer.weight,
                        -3.0 / math.sqrt(current_size + n_classes),
                        3.0 / math.sqrt(current_size + n_classes))
        torch.nn.init.zeros_(self.output_layer.bias)
        
        # 存储输出层参数
        self.params["W_out"] = self.output_layer.weight
        self.params["b_out"] = self.output_layer.bias
    
    def forward(self, x, dropout_rates=None):
        # 存储每层激活值
        activations = [x]
        
        # 如果提供了dropout率，则使用提供的值
        if dropout_rates is not None:
            for i, dropout in enumerate(self.dropouts):
                if i < len(dropout_rates):
                    # 确保是保留率到丢弃率的正确转换
                    keep_prob = dropout_rates[i]
                    dropout.p = 1.0 - keep_prob  # 注意：TF中是保留率，而PyTorch中是丢弃率
        
        # 前向传播
        for module in self.layer_modules:
            x = module(x)
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Tanh) or isinstance(module, torch.nn.Sigmoid) or isinstance(module, torch.nn.ReLU):
                activations.append(x)
        
        # 输出层
        y_hat = self.output_layer(x)
        activations.append(y_hat)
        
        return y_hat, F.softmax(y_hat, dim=1), activations
    
    def get_cost(self, y_hat, y):
        return F.cross_entropy(y_hat, y)
    
    def get_params(self):
        return self.params


# 为了兼容性，提供与原始接口类似的函数
def ae(input_size, code_size, corruption=0.0, tight=False, enc=torch.tanh, dec=torch.tanh):
    return AutoEncoder(input_size, code_size, corruption, tight, enc, dec)


def nn(input_size, n_classes, layers, init=None):
    return MultiLayerNN(input_size, n_classes, layers, init)