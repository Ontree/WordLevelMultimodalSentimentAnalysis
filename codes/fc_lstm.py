import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tools.data_loader as loader
import numpy as np
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from fc_model import FCLSTM
from math import floor
np.random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch FC-LSTM Model for Word Level Sentiment Analysis')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--max_segment_len', default=115, type=int, help='')
parser.add_argument('-s', '--feature_selection', default=0, type=int, choices=[0,1], help='whether to use feature_selection')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--nlayers', type=int, default=1, metavar='N',
                    help='num of LSTM layers')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='report interval')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


#----------lOAD DATA-------
tr_split = 2.0/3                        # fixed. 62 training & validation, 31 test
val_split = 0.1514                      # fixed. 52 training 10 validation
use_pretrained_word_embedding = True    # fixed. use glove 300d
embedding_vecor_length = 300            # fixed. use glove 300d
max_segment_len = args.max_segment_len  #115                   # fixed for MOSI. The max length of a segment in MOSI dataset is 114
end_to_end = True                       # fixed
lstm_units = 64
word_embedding = loader.load_word_embedding()
train, test = loader.load_word_level_features(max_segment_len, tr_split)
feature_str = ''
if args.feature_selection:
    with open('/usr0/home/minghai1/777/preprocess/fs_mask.pkl') as f:
        [covarep_ix, facet_ix] = pickle.load(f)
    facet_train = train['facet'][:,:,facet_ix]
    facet_test = test['facet'][:,:,facet_ix]
    covarep_train = train['covarep'][:,:,covarep_ix]
    covarep_test = test['covarep'][:,:,covarep_ix]
    feature_str = '_c'+str(covarep_test.shape[2]) + '_f'+str(facet_test.shape[2])
else:
    facet_train = train['facet']
    covarep_train = train['covarep'][:,:,1:35]
    facet_test = test['facet']
    covarep_test = test['covarep'][:,:,1:35]
text_train = train['text']
text_test = test['text']
y_train = train['label']
y_train = y_train.reshape((len(y_train),1))
y_test = test['label']
y_test = y_test.reshape((len(y_test),1))
facet_train_max = np.max(np.max(np.abs(facet_train ), axis =0),axis=0)
facet_train_max[facet_train_max==0] = 1
covarep_train_max =  np.max(np.max(np.abs(covarep_train), axis =0),axis=0)
covarep_train_max[covarep_train_max==0] = 1

facet_train = facet_train / facet_train_max
facet_test = facet_test / facet_train_max
covarep_train = covarep_train / covarep_train_max
covarep_test = covarep_test / covarep_train_max
embedding_train = np.zeros((text_train.shape[0],text_train.shape[1],len(word_embedding[0])))
for i in range(text_train.shape[0]):
    for j in range(text_train.shape[1]):
        embedding_train[i][j] = word_embedding[text_train[i][j]]


data_size = embedding_train.shape[0]
valid_size = int(val_split*data_size)
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
y_valid = y_train[-data_size:]
y_valid = torch.from_numpy(y_valid)
y_train = y_train[:-data_size]
y_train = torch.from_numpy(y_train)
if args.cuda:
    embedding_train.cuda()
    embedding_valid.cuda()
    facet_train.cuda()
    facet_valid.cuda()
    covarep_train.cuda()
    covarep_valid.cuda()
    y_train.cuda()
    y_valid.cuda()
# X_train, X_test = [text_train], [text_test]
# X_train.append(covarep_train)
# X_test.append(covarep_test)
# X_train.append(facet_train)
# X_test.append(facet_test)
#--------END LOAD DATA

batch_size = args.batch_size
nlayers = args.nlayers
dropout = args.dropout

criterion = nn.L1Loss()
model = FCLSTM(embedding_train.size(1),embedding_train.size(2),facet_train.size(2),covarep_train.size(2),lstm_units,batch_size,nlayers,dropout)
if args.cuda:
    model.cuda()
def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)
def get_batch(t_data,v_data,a_data,y,ix,batch_size,evaluation = False):
    return [[Variable(t_data[ix*batch_size:ix*(batch_size+1)],volatile=evaluation),Variable(v_data[ix*batch_size:ix*(batch_size+1)],volatile=evaluation),Variable(a_data[ix*batch_size,(ix+1)*batch_size],volatile=evaluation)],Variable(y[ix*batch_size:(ix+1)*batch_size])]

def evaluate(iterations):
    model.eval()
    total_loss = 0
    for i in xrange(iterations):
        input, target = get_batch(embedding_valid,facet_valid,covarep_valid,y_valid,i,batch_size)
        loss = criterion(target, output)
        total_loss += loss.data
    return total_loss / iterations
def train(iterations,lr,epoch):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    total_loss = 0
    start_time = time.time()
    for i in xrange(iterations):
        input, target = get_batch(embedding_train,facet_train,covarep_train,y_train,i,batch_size, evaluation = True)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(target,output)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        total_loss += loss.data
        optimizer.step()
        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f}'.format(
                epoch, i, iterations, lr,
                elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()
lr = args.lr
best_val_loss = None
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_iterations = int(floor(len(embedding_train) / batch_size))
        valid_iterations = int(floor(len(embedding_valid) / batch_size))
        train(train_iterations,lr,epoch)
        val_loss = evaluate(valid_iterations)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
