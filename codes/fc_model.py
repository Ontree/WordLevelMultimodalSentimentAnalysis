import torch.nn as nn
from torch.autograd import Variable
import math
class FCLSTM(nn.Module):
    def __init__(self, seq_len, text_size, visual_size, acc_size, hidden_size, batch_size, nlayers, dropout=0.5):
        super(FCLSTM, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(3*hidden_size, 1)
        self.TLSTM = nn.LSTM(text_size+2*hidden_size, hidden_size, nlayers, dropout = dropout)
        self.VLSTM = nn.LSTM(visual_size+2*hidden_size, hidden_size, nlayers, dropout=dropout)
        self.ALSTM = nn.LSTM(acc_size+2*hidden_size, hidden_size, nlayers, dropout=dropout)
        self.init_weights()
    def xavier_normal(self, tensor, gain=1):
        if isinstance(tensor, Variable):
            self.xavier_normal(tensor.data, gain=gain)
            return tensor
        fan_in, fan_out = tensor.size(0), tensor.size(1)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        return tensor.normal_(0, std)
    def init_weights(self):
        #initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data = self.xavier_normal(self.decoder.weight.data)

    def forward(self,input):
        t_input = input[0]
        v_input = input[1]
        a_input = input[2]
        hti = self.init_hidden(self.batch_size)
        hvi = self.init_hidden(self.batch_size)
        hai = self.init_hidden(self.batch_size)
        hti_feature = hti[0][-1, :, :]
        hti_feature = hti_feature.unsqueeze(1)
        hvi_feature = hvi[0][-1, :, :]
        hvi_feature = hvi_feature.unsqueeze(1)
        hai_feature = hai[0][-1, :, :]
        hai_feature = hai_feature.unsqueeze(1)
        for i in range(self.seq_len):
            it_input = t_input[:,i,:].contiguous()
            it_input = it_input.unsqueeze(1)
            it_input = self.drop(torch.cat((it_input,hvi_feature,hai_feature),2))
            iv_input = v_input[:, i, :].contiguous()
            iv_input = iv_input.unsqueeze(1)
            iv_input = self.drop(torch.cat((iv_input,hti_feature,hai_feature),2))
            ia_input = a_input[:, i, :].contiguous()
            ia_input = ia_input.unsqueeze(1)
            ia_input = self.drop(torch.cat((ia_input,hti_feature,hvi_feature),2))
            # concatenate input with features
            _, hti = self.TLSTM(it_input, hti)
            _, hvi = self.VLSTM(iv_input,hvi)
            _, hai = self.ALSTM(ia_input, hai)
            hti_feature = hti[0][-1,:,:]
            hti_feature = hti_feature.unsqueeze(1)
            hvi_feature = hvi[0][-1,:,:]
            hvi_feature = hvi_feature.unsqueeze(1)
            hai_feature = hai[0][-1,:,:]
            hai_feature = hai_feature.unsqueeze(1)
        hti = hti[0][-1,:,:]
        hvi = hvi[0][-1,:,:]
        hai = hai[0][-1,:,:]
        hidden_feature = self.drop(torch.cat((hti,hvi,hai), 1))
        output = self.decoder(hidden_feature)
        return output
    def init_hidden(self,batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_()))

