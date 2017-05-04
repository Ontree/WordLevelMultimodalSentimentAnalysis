import cPickle as cp
import tools.data_loader as loader
word2ix = loader.load_word2ix()
max_segment_len = 115
tr_split = 2.0/3
train, test = loader.load_word_level_features(max_segment_len, tr_split)
ix2word = {word2ix[x]: x for x in word2ix}
ix2word[0] = ''
text_test = test['text']
text = [''.join([ix2word[x] for x in text_test[i,:]]) for i in range(text_test.shape[0])]
rl_predictions, masks = cp.load(open('rl_results.pkl'))
conv_predictions, true_y, ids = cp.load(open('conv_results.pkl'))
l = []
for i in range(len(true_y)):
    l.append([abs(abs(true_y[i]-conv_predictions[i])-abs(true_y[i]-rl_predictions[i])),rl_predictions[i],conv_predictions[i],true_y[i],masks[i],ids[i],text[i]])
l = sorted(l, key = lambda a:a[0])
for i in range(10):
    a = l[-10+i]
    print a[0][0]
    print a[1][0]
    print a[2]
    print a[3]
    print a[4]
    print a[5]
    print a[6]
    print '='*89