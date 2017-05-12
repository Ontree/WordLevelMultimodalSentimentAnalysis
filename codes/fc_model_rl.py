import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tools.data_loader as loader
import numpy as np
import argparse
import time
import math
import torch
import torch.nn as nn
import cPickle as pickle
from torch.autograd import Variable
import torch.optim as optim
from fc_model_mean import FCLSTM
from math import floor
np.random.seed(0)
seed = 1111
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

class FC_Model():
    def __init__(self,text_hidden_size=64,visual_hidden_size=8,acc_hidden_size=8):
        self.nlayers = 1
        self.dropout = 0.5
        self.batch_size = 20
        self.lr = 0.0005
        self.num_epochs = 200
        self.clip = 0.25
        self.log_interval = 5
        self.visual_hidden_size = visual_hidden_size
        self.text_hidden_size = text_hidden_size
        self.acc_hidden_size = acc_hidden_size
        self.save = 'fclstm.'+'_'.join(str(self.nlayers),str(self.dropout),str(self.batch_size),str(self.lr),str(self.num_epochs))+'.result'
        self.embedding_train, self.facet_train, self.covarep_train, self.y_train, self.embedding_valid, self.facet_valid, self.covarep_valid,self.y_valid,self.embedding_test,self.facet_test,self.covarep_test,self.y_test = self.load_data()
        self.criterion = nn.L1Loss()
        self.model = FCLSTM( self.embedding_train.size(1),  self.embedding_train.size(2),  self.facet_train.size(2),  self.covarep_train.size(2),
                             text_hidden_size,  visual_hidden_size, acc_hidden_size, self.batch_size, self.nlayers, self.dropout)
        if torch.cuda.is_available():
            self.model.cuda()

    def get_batch(self,t_data, v_data, a_data, y, ix, batch_size, evaluation=False):
        return [[Variable(t_data[ix * batch_size:(ix + 1) * batch_size].cuda(), volatile=evaluation),
                 Variable(v_data[ix * batch_size:(ix + 1) * batch_size].cuda(), volatile=evaluation),
                 Variable(a_data[ix * batch_size:(ix + 1) * batch_size].cuda(), volatile=evaluation)],
                Variable(y[ix * batch_size:(ix + 1) * batch_size].cuda())]

    def train(self, iterations, lr, epoch, embedding_train, facet_train, covarep_train, y_train):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        data_size = embedding_train.size(0)
        ixs = torch.randperm(data_size)
        embedding_train = embedding_train[ixs]
        facet_train = facet_train[ixs]
        covarep_train = covarep_train[ixs]
        y_train = y_train[ixs]
        total_loss = 0
        start_time = time.time()
        for i in xrange(iterations):
            input, target = self.get_batch(embedding_train, facet_train, covarep_train, y_train, i, self.batch_size)
            optimizer.zero_grad()
            output = self.model(input)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
            total_loss += loss.data
            optimizer.step()
            if i % self.log_interval == 0 and i > 0:
                cur_loss = total_loss[0] / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                      'loss {:5.2f}'.format(
                    epoch, i, iterations, lr,
                    elapsed * 1000 / self.log_interval, cur_loss))
                total_loss = 0
                start_time = time.time()

    def test(self):
        test_iterations = int(floor(len(self.embedding_test) / self.batch_size))
        self.model.eval()
        total_loss = 0
        acc = 0
        for i in xrange(test_iterations):
            input, target = self.get_batch(self.embedding_test, self.facet_test, self.covarep_test, self.y_test, i, batch_size, evaluation=True)
            output = self.model(input)
            loss = self.criterion(target, output)
            total_loss += loss.data
            output_data = output.data.numpy()
            target_data = target.data.numpy()
            acc_2 = sum((output_data < 0) == (target_data < 0)) / float(len(output_data))
            acc += acc_2
        return [total_loss[0] / test_iterations, acc / test_iterations]


    def evaluate(self,iterations):
        self.model.eval()
        total_loss = 0
        acc = 0
        for i in xrange(iterations):
            input, target = self.get_batch(self.embedding_valid, self.facet_valid,self.covarep_valid, self.y_valid, i, batch_size,
                                      evaluation=True)
            output = self.model(input)
            loss = self.criterion(target, output)
            output_data = output.data.numpy()
            target_data = target.data.numpy()
            acc_2 = sum((output_data < 0) == (target_data < 0)) / float(len(output_data))
            total_loss += loss.data
            acc += acc_2
        return [total_loss[0] / iterations, acc / iterations]
    
    def reset(self):
        self.model = FCLSTM( self.embedding_train.size(1),  self.embedding_train.size(2),  self.facet_train.size(2),  self.covarep_train.size(2),
                             self.text_hidden_size,  self.visual_hidden_size, self.acc_hidden_size, self.batch_size, self.nlayers, self.dropout)
        if torch.cuda.is_available():
            self.model.cuda()
    def save_weight(self,path):
        with open(path, 'wb') as f:
            torch.save(self.model, f)
    def load_weight(self,path):
        with open(path, 'rb') as f:
            self.model = torch.load(f)
    def fit(self):
        history = []
        best_val_loss = None
        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()
            train_iterations = int(floor(len(self.embedding_train) / self.batch_size))
            valid_iterations = int(floor(len(self.embedding_valid) / self.batch_size))
            self.train(train_iterations, self.lr, epoch, self.embedding_train, self.facet_train, self.covarep_train, self.y_train)
            record = self.evaluate(valid_iterations)
            history.append(record)
            val_loss = record[0]
            print('-' * 89)
            print(
            '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                                                val_loss))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(self.save, 'wb') as f:
                    torch.save(self.model, f)
                best_val_loss = val_loss
            print 'record:'
            print record
        return history
    def load_data(self):
        tr_split = 2.0 / 3  # fixed. 62 training & validation, 31 test
        val_split = 0.1514  # fixed. 52 training 10 validation
        use_pretrained_word_embedding = True  # fixed. use glove 300d
        embedding_vecor_length = 300  # fixed. use glove 300d
        max_segment_len = 115  # 115                   # fixed for MOSI. The max length of a segment in MOSI dataset is 114
        end_to_end = True  # fixed
        text_hidden_size = 64
        visual_hidden_size = 8
        acc_hidden_size = 8
        word_embedding = loader.load_word_embedding()
        train, test = loader.load_word_level_features(max_segment_len, tr_split)
        feature_str = ''
        if True:
            with open('fs_mask.pkl') as f:
                [covarep_ix, facet_ix] = pickle.load(f)
            facet_train = train['facet'][:, :, facet_ix]
            facet_test = test['facet'][:, :, facet_ix]
            covarep_train = train['covarep'][:, :, covarep_ix]
            covarep_test = test['covarep'][:, :, covarep_ix]
            feature_str = '_c' + str(covarep_test.shape[2]) + '_f' + str(facet_test.shape[2])
        else:
            facet_train = train['facet']
            covarep_train = train['covarep'][:, :, 1:35]
            facet_test = test['facet']
            covarep_test = test['covarep'][:, :, 1:35]
        text_train = train['text']
        text_test = test['text']
        y_train = train['label']
        y_train = y_train.reshape((len(y_train), 1))
        y_test = test['label']
        y_test = y_test.reshape((len(y_test), 1))
        facet_train_max = np.max(np.max(np.abs(facet_train), axis=0), axis=0)
        facet_train_max[facet_train_max == 0] = 1
        covarep_train_max = np.max(np.max(np.abs(covarep_train), axis=0), axis=0)
        covarep_train_max[covarep_train_max == 0] = 1
        facet_train = facet_train / facet_train_max
        facet_test = facet_test / facet_train_max
        covarep_train = covarep_train / covarep_train_max
        covarep_test = covarep_test / covarep_train_max
        embedding_train = np.zeros((text_train.shape[0], text_train.shape[1], len(word_embedding[0])))
        embedding_test = np.zeros((text_test.shape[0], text_test.shape[1], len(word_embedding[0])))

        for i in range(text_train.shape[0]):
            for j in range(text_train.shape[1]):
                embedding_train[i][j] = word_embedding[text_train[i][j]]

        for i in range(text_test.shape[0]):
            for j in range(text_test.shape[1]):
                embedding_test[i][j] = word_embedding[text_test[i][j]]

        data_size = embedding_train.shape[0]
        valid_size = int(val_split * data_size)
        embedding_valid = embedding_train[-valid_size:]
        embedding_valid = torch.from_numpy(embedding_valid)
        facet_valid = facet_train[-valid_size:]
        facet_valid = torch.from_numpy(facet_valid)
        covarep_valid = covarep_train[-valid_size:]
        covarep_valid = torch.from_numpy(covarep_valid)
        embedding_train = embedding_train[:-valid_size]
        embedding_train = torch.from_numpy(embedding_train)
        facet_train = facet_train[:-valid_size]
        facet_train = torch.from_numpy(facet_train)
        covarep_train = covarep_train[:-valid_size]
        covarep_train = torch.from_numpy(covarep_train)
        y_valid = y_train[-valid_size:]
        y_valid = torch.from_numpy(y_valid).float()
        y_train = y_train[:-valid_size]
        y_train = torch.from_numpy(y_train).float()
        embedding_test = torch.from_numpy(embedding_test).float()
        facet_test = torch.from_numpy(facet_test).float()
        covarep_test = torch.from_numpy(covarep_test).float()
        y_test = torch.from_numpy(y_test).float()
        return embedding_train,facet_train,covarep_train,y_train,embedding_valid,facet_valid,covarep_valid,y_valid,embedding_test,facet_test,covarep_test,y_test