import torch.nn as nn
from torch.autograd import Variable

class FCLSTM(nn.Module):
    def __init__(self, text_size, visual_size, acc_size, hidden_size, nlayers, dropout=0.5):
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(3*hidden_size, 1)
        self.TLSTM = nn.LSTM(text_size+2*hidden_size, hidden_size, nlayers, dropout = dropout)
        self.VLSTM = nn.LSTM(visual_size+2*hidden_size, hidden_size, nlayers, dropout=dropout)
        self.ALSTM = nn.LSTM(acc_size+2*hidden_size, hidden_size, nlayers, dropout=dropout)
        self.init_weights()
    def init_weights(self):

    def forward(self,input):

    def init_hidden(self):