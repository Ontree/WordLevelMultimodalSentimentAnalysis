import torch.nn as nn
import torch
from torch.autograd import Variable
import math
class FCLSTM(nn.Module):
    def __init__(self, seq_len, text_size, visual_size, acc_size, text_hidden_size, visual_hidden_size, acc_hidden_size, batch_size, nlayers, dropout=0.5):
        super(FCLSTM, self).__init__()
        self.text_hidden_size = text_hidden_size
        self.visual_hidden_size = visual_hidden_size
        self.acc_hidden_size = acc_hidden_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(text_hidden_size+visual_hidden_size+acc_hidden_size, 1)
        self.nlayers = nlayers
        self.TLSTM = nn.LSTM(text_size+visual_hidden_size+acc_hidden_size, text_hidden_size, nlayers, dropout = dropout, batch_first = True,bidirectional=True)
        self.VLSTM = nn.LSTM(visual_size+text_hidden_size+acc_hidden_size, visual_hidden_size, nlayers, dropout=dropout, batch_first = True,bidirectional=True)
        self.ALSTM = nn.LSTM(acc_size+text_hidden_size+visual_hidden_size, acc_hidden_size, nlayers, dropout=dropout, batch_first = True,bidirectional=True)
        self.bidirectional = True
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
        hti = self.init_hidden(self.batch_size,self.text_hidden_size)
        hvi = self.init_hidden(self.batch_size,self.visual_hidden_size)
        hai = self.init_hidden(self.batch_size,self.acc_hidden_size)
        hti_feature = hti[0][-1, :, :]
        t_final_feature = Variable(hti_feature.data)
        hti_feature = hti_feature.unsqueeze(1)
        hvi_feature = hvi[0][-1, :, :]
        v_final_feature = Variable(hvi_feature.data)
        hvi_feature = hvi_feature.unsqueeze(1)
        hai_feature = hai[0][-1, :, :]
        a_final_feature = Variable(hai_feature.data)
        hai_feature = hai_feature.unsqueeze(1)
        for i in range(self.seq_len):
            it_input = t_input[:,i,:].contiguous().float()
            it_input = it_input.unsqueeze(1)
            it_input = self.drop(torch.cat((it_input,hvi_feature,hai_feature),2))
            iv_input = v_input[:, i, :].contiguous().float()
            iv_input = iv_input.unsqueeze(1)
            iv_input = self.drop(torch.cat((iv_input,hti_feature,hai_feature),2))
            ia_input = a_input[:, i, :].contiguous().float()
            ia_input = ia_input.unsqueeze(1)
            ia_input = self.drop(torch.cat((ia_input,hti_feature,hvi_feature),2))
            # concatenate input with features
            _, hti = self.TLSTM(it_input, hti)
            _, hvi = self.VLSTM(iv_input,hvi)
            _, hai = self.ALSTM(ia_input, hai)
            hti_feature = hti[0][-1,:,:]
            t_final_feature = t_final_feature + hti_feature
            hti_feature = hti_feature.unsqueeze(1)
            hvi_feature = hvi[0][-1,:,:]
            v_final_feature = v_final_feature + hvi_feature
            hvi_feature = hvi_feature.unsqueeze(1)
            hai_feature = hai[0][-1,:,:]
            a_final_feature = a_final_feature + hai_feature
            hai_feature = hai_feature.unsqueeze(1)
        t_final_feature = t_final_feature/float(self.seq_len)
        v_final_feature = v_final_feature/float(self.seq_len)
        a_final_feature = a_final_feature/float(self.seq_len)
        hidden_feature = self.drop(torch.cat((t_final_feature,v_final_feature,a_final_feature), 1))
        output = self.decoder(hidden_feature)
        return output
    def init_hidden(self,batch_size,hidden_size):
        weight = next(self.parameters()).data
        if self.bidirectional:
            par = 2
        else:
            par = 1
        return (Variable(weight.new(par*self.nlayers, batch_size, hidden_size).zero_()),
                    Variable(weight.new(par*self.nlayers, batch_size, hidden_size).zero_()))

