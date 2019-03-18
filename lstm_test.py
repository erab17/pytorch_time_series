# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 08:15:58 2019

@author: e070632
"""

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

a = torch.Tensor([1,2,3]).numpy()

x = np.linspace(1,40,10000)

inputData = np.sin(x)
inputData.reshape(-1, 1)
b = "test"
print(a)

sns.tsplot(inputData)

seq_length = 40
n = 0

batches_n = 80
batch_data_x = np.zeros((batches_n, seq_length))
batch_data_y = np.zeros((batches_n, 1))

for x in range(0, batches_n, 1):
    #print(x)
    
    batch_row_x = inputData[x:x + seq_length]
    batch_data_x[n,:] = batch_row_x
    
    batch_data_y[n, 0] = inputData[x + seq_length + 1]
    if n > batches_n:
        break
    n += 1
    
    
batch_size = 10
# create Tensor datasets
data = TensorDataset(torch.from_numpy(batch_data_x), torch.from_numpy(batch_data_y))
# make sure to SHUFFLE your training data
data_loader = DataLoader(data, shuffle=False, batch_size=batch_size)



class RNN(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout=0.0):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()       
        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        #self.sig = nn.Sigmoid()
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        batch_size = nn_input.size(0)

        #SPECIAL
        lstm_out, hidden = self.lstm(nn_input, hidden)
        
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        #out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        # sigmoid function
        #sig_out = self.sig(out)
        
        # reshape into (batch_size, seq_length, output_size)
        out = out.view(batch_size, -1, self.output_size)
        # get last batch
        out = out[:, -1]
    
    
        # return last sigmoid output and hidden state
        #return sig_out, hidden
        return out, hidden
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        
        # initialize hidden state with zero weights, and move to GPU if available
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden
    
model = RNN(1, 1, 5, 1)
