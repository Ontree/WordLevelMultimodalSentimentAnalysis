import cPickle as cp
rl_predictions, masks = cp.load(open('rl_results.pkl'))
conv_predictions, true_y, ids = cp.load(open('conv_results.pkl'))
l = []
for i in range(len(true_y)):
    l.append([abs(abs(true_y[i]-conv_predictions[i])-abs(true_y[i]-rl_predictions[i])),rl_predictions[i],conv_predictions[i],true_y[i],masks[i],ids[i]])
l = sorted(l, key = lambda a:a[0])
print l[-10:]