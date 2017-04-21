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
np.random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch FC-LSTM Model for Word Level Sentiment Analysis')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--max_segment_len', default=115, type=int, help='')
parser.add_argument('-s', '--feature_selection', default=0, type=int, choices=[0,1], help='whether to use feature_selection')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
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
word_embedding = [loader.load_word_embedding()] if use_pretrained_word_embedding else None
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
y_test = test['label']

facet_train_max = np.max(np.max(np.abs(facet_train ), axis =0),axis=0)
facet_train_max[facet_train_max==0] = 1
#covarep_train_max =  np.max(np.max(np.abs(covarep_train), axis =0),axis=0)
#covarep_train_max[covarep_train_max==0] = 1

facet_train = facet_train / facet_train_max
facet_test = facet_test / facet_train_max
covarep_train = covarep_train / covarep_train_max
covarep_test = covarep_test / covarep_train_max
# X_train, X_test = [text_train], [text_test]
# X_train.append(covarep_train)
# X_test.append(covarep_test)
# X_train.append(facet_train)
# X_test.append(facet_test)
#--------END LOAD DATA


if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
criterion = nn.L1Loss()

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)



if __name__ == '__main__':
