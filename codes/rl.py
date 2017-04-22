import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tools.data_loader as loader
import numpy as np
np.random.seed(0)
import tensorflow as tf
tf.set_random_seed(0)
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Flatten, Activation, Permute, RepeatVector, Lambda
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
from keras.layers.merge import Multiply
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.regularizers import l2
from keras.layers import merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import keras.backend as K
import argparse
from keras.layers.wrappers import TimeDistributed
import cPickle as pickle
import controller



def get_rl_x_test(X_test_ori, f_controller, max_segment_len):
    facet_test = X_test_ori[-1]
    facet_test_2d = np.reshape(facet_test, [-1, facet_test.shape[-1]])
    facet_test_mask_p = f_controller.predict(facet_test_2d)
    facet_mask_test = []
    for k in range(facet_test_mask_p.shape[0]):
        p_0 = facet_test_mask_p[k, 0]
        if p_0 >= 0.5:
            facet_mask_test.append(0)
        else:
            facet_mask_test.append(1)
    facet_mask_test=np.array(facet_mask_test)
    facet_mask_test.shape = (-1, max_segment_len, 1)
    X_test = X_test_ori[:-1] + [X_test_ori[-1] * facet_mask_test]
    return X_test


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if isinstance(layer, Dense) or isinstance(layer, LSTM):
            old = layer.get_weights()
            for w in layer._trainable_weights:
                w.initializer.run(session=session)
            print(np.array_equal(old, layer.get_weights())," after initializer run")
        else:
            pass
            #print(layer, "not reinitialized")


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
K.get_session().run(tf.initialize_all_variables())


parser = argparse.ArgumentParser(description='')
parser.add_argument('--train_epoch', default=1000, type=int)
parser.add_argument('--train_patience', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--pretrain', default=0, type=int, choices=[0, 1], help='0: use all the modalities; 1: use only text, other modalities are set as 0')
parser.add_argument('-f', '--feature', default=['f','c'], type=list, help='what features to use besides text. c: covarep; f: facet. default is null')
parser.add_argument('-a', '--attention', default=1, type=int, choices=[0,1], help='whether to use attention model. 1: use; 0: not. default is 0')
parser.add_argument('-s', '--feature_selection', default=1, type=int, choices=[0,1], help='whether to use feature_selection')
parser.add_argument('-c', '--convolution', default=0, type=int, choices=[0,1], help='whether to use convolutional layer on covarep and facet')
parser.add_argument('--max_segment_len', default=115, type=int, help='')
parser.add_argument('--rl_sample_n', default=10, type=int)
parser.add_argument('--rl_all_epoch', default=100, type=int)
parser.add_argument('-r', '--rl', default=1, type=int, choices=[0, 1], help='1: use rl')

args = parser.parse_args()


tr_split = 2.0/3                        # fixed. 62 training & validation, 31 test
val_split = 0.1514                      # fixed. 52 training 10 validation
use_pretrained_word_embedding = True    # fixed. use glove 300d
embedding_vecor_length = 300            # fixed. use glove 300d
max_segment_len = args.max_segment_len  #115                   # fixed for MOSI. The max length of a segment in MOSI dataset is 114 
end_to_end = True                       # fixed
lstm_units = 64

#word2ix = loader.load_word2ix()
word_embedding = [loader.load_word_embedding()] if use_pretrained_word_embedding else None
train, test = loader.load_word_level_features(max_segment_len, tr_split)

feature_str = ''
if args.feature_selection:
    with open('/usr0/home/minghai1/multimodal/preprocess/fs_mask.pkl') as f:
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

train_n = train['text'].shape[0]
test_n = test['text'].shape[0]
facet_train_max = np.max(np.max(np.abs(facet_train ), axis =0),axis=0)
facet_train_max[facet_train_max==0] = 1
#covarep_train_max =  np.max(np.max(np.abs(covarep_train), axis =0),axis=0)
#covarep_train_max[covarep_train_max==0] = 1

facet_train = facet_train / facet_train_max
#covarep_train = covarep_train / covarep_train_max
facet_test = facet_test / facet_train_max
#covarep_test = covarep_test / covarep_train_max


weights_folder_path = '../weights/'
weights_path = weights_folder_path + ''.join(sorted(args.feature)) + ("_attention" if args.attention else "") +  feature_str +"-rl.h5"


if args.pretrain:
    # use only text to pretrain the model
    facet_test, facet_train = np.zeros(facet_test.shape), np.zeros(facet_train.shape)
    covarep_test, covarep_train = np.zeros(covarep_test.shape), np.zeros(covarep_train.shape)

X_train, X_test = [text_train], [text_test]
if 'c' in args.feature:
    X_train.append(covarep_train)
    X_test.append(covarep_test)
if 'f' in args.feature:
    X_train.append(facet_train)
    X_test.append(facet_test)


facet_dim = facet_train.shape[2]
text_input = Input(shape=(max_segment_len,), dtype='int32', name='text_input')
text_eb_layer = Embedding(word_embedding[0].shape[0], embedding_vecor_length, input_length=max_segment_len, weights=word_embedding, name = 'text_eb_layer', trainable=False)(text_input)
facet_input = Input(shape=(max_segment_len, facet_dim), name='facet_input')
covarep_input = Input(shape=(max_segment_len, covarep_train.shape[2]), name='covarep_input')

# convolutional layers
if args.convolution:
    facet_layer = Conv1D(facet_train.shape[2], 3, padding='same', activation='tanh')(facet_input)
    covarep_layer = Conv1D(covarep_train.shape[2], 3, padding='same', activation='tanh')(covarep_input)
else:
    facet_layer = facet_input
    covarep_layer = covarep_input

unimodel_layers = [text_eb_layer]
model_input = [text_input]
if 'c' in args.feature:
    unimodel_layers.append(covarep_layer)
    model_input.append(covarep_input)
if 'f' in args.feature:
    model_input.append(facet_input)
    unimodel_layers.append(facet_layer)

if len(unimodel_layers) > 1:
    merge_input = merge(unimodel_layers,  mode='concat')
else:
    merge_input = text_eb_layer




if args.attention:
    activations = LSTM(lstm_units, name = 'lstm_layer', trainable=end_to_end, return_sequences=True)(merge_input)
    attention = TimeDistributed(Dense(1, activation='tanh'))(activations) 
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(lstm_units)(attention)
    attention = Permute([2, 1])(attention)

    # apply the attention
    sent_representation = merge([activations, attention], mode='mul')
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
else:
    sent_representation = LSTM(64, name = 'lstm_layer', trainable=end_to_end)(merge_input)

output_layer_1 = Dense(1, name = 'dense_layer', W_regularizer=l2(0.01))(sent_representation)



callbacks = [
    EarlyStopping(monitor='val_loss', patience=args.train_patience, verbose=0),
    ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=0),
]

#sgd = SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
model = Model(model_input, output_layer_1)
adam_model = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mae', optimizer=adam_model)
print(model.summary())


if args.rl and 'f' in args.feature:
    min_loss = 99999
    X_train_ori, X_test_ori = X_train, X_test
    f_controller =  controller.visual_controller(facet_dim)
    adam_controller = optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    f_controller.compile(loss=controller.my_loss, optimizer=adam_controller)

    facet_train = X_train_ori[-1]
    facet_train_2d = np.reshape(facet_train, [-1, facet_train.shape[-1]])
    base_loss = -999
    for i in range(args.rl_all_epoch):
        facet_train_mask_p = f_controller.predict(facet_train_2d)
        facet_controller_train_X = [] 
        facet_controller_train_y = []
        sample_losses = []
        for j in range(args.rl_sample_n):
            print('i:'+str(i)+' j:'+str(j))
            facet_mask_train = []
            facet_controller_train_y_part = []
            for k in range(facet_train_mask_p.shape[0]):
                if i == 0:
                    p_0 = 0.5
                else:
                    p_0 = facet_train_mask_p[k, 0]
                rand = np.random.random()
                if sum(facet_train_2d[k]) != 0:     
                    facet_controller_train_X.append(facet_train_2d[k])     
                    if rand < p_0:
                        facet_mask_train.append(0)
                        facet_controller_train_y_part.append([1, 0])
                    else:
                        facet_mask_train.append(1)
                        facet_controller_train_y_part.append([0, 1])
                else:
                    facet_mask_train.append(0)
            facet_mask_train=np.array(facet_mask_train)
            facet_mask_train.shape = (-1, max_segment_len, 1)
            X_train = X_train_ori[:-1] + [X_train_ori[-1] * facet_mask_train]#[:, :, np.newaxis]]
            #X_test = X_test_ori[:-1] + [X_test_ori[-1] * facet_mask_test[:, :, np.newaxis]]
            reset_weights(model)
            adam_model = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            model.compile(loss='mae', optimizer=adam_model)
            history = model.fit(X_train, y_train, validation_split=val_split, nb_epoch=args.train_epoch, batch_size=args.batch_size, callbacks=callbacks)
            #history = model.fit(X_train, y_train, validation_split=val_split, nb_epoch=3, batch_size=args.batch_size, callbacks=callbacks)
            loss = min(history.history['val_loss'])
            if min_loss > loss:
                min_loss = loss
                model.load_weights(weights_path)
                model.save_weights("../weights/rl_model.h5")
                f_controller.save_weights('../weights/controller.h5')
                X_test = get_rl_x_test(X_test_ori, f_controller, max_segment_len)
                mae = model.evaluate(X_test, y_test, verbose=0)
                with open('mae.txt', 'a') as f:
                    f.write(str(loss) + '  ' + str(mae)+'\n')
            sample_losses.append(loss)
            if base_loss == -999:
                base_loss = loss
            facet_controller_train_y_part = np.array(facet_controller_train_y_part)
            facet_controller_train_y_part = facet_controller_train_y_part * (loss - base_loss)
            facet_controller_train_y.append(facet_controller_train_y_part)
        sample_loss_mean = np.mean(sample_losses)
        base_loss = 0.2*base_loss + (1-0.2)*sample_loss_mean
        facet_controller_train_y = np.concatenate(facet_controller_train_y)
        facet_controller_train_X = np.array(facet_controller_train_X)
        f_controller.fit(facet_controller_train_X, facet_controller_train_y, nb_epoch=50)#, batch_size=1000)
    '''
    for i in range(100000):
        f_controller.fit(facet_controller_train_X, facet_controller_train_y, nb_epoch=1, batch_size=4000)
        aaa = np.min(f_controller.predict(facet_controller_train_X))
        print aaa, np.log(aaa)
    '''
    '''
    facet_test = X_test_ori[-1]
    facet_test_2d = np.reshape(facet_test, [-1, facet_test.shape[-1]])
    facet_test_mask_p = f_controller.predict(facet_test_2d)
    facet_mask_test = []
    for k in range(facet_test_mask_p.shape[0]):
        p_0 = facet_test_mask_p[k, 0]
        if p_0 >= 0.5:
            facet_mask_test.append(0)
        else:
            facet_mask_test.append(1)
    facet_mask_test=np.array(facet_mask_test)
    facet_mask_test.shape = (-1, max_segment_len, 1)
    X_test = X_test_ori[:-1] + [X_test_ori[-1] * facet_mask_test]
    '''
    X_test = get_rl_x_test(X_test_ori, f_controller, max_segment_len)

# Final evaluation of the model
model.load_weights(weights_path)
mae = model.evaluate(X_test, y_test, verbose=0)
predictions = model.predict(X_test)
predictions = predictions.reshape(-1,)
cor = np.corrcoef(predictions, y_test)[0, 1]
ccc = (2 * np.cov(np.array([predictions, y_test]), bias = True) / (np.var(predictions) + np.var(y_test) + (np.mean(predictions) - np.mean(y_test)) ** 2))[0][1]
acc_7 = sum(np.round(predictions)==np.round(y_test))/float(len(y_test))
#acc_5 = sum(tool.score2label_7(predictions)==tool.score2label_7(y_test))/float(len(y_test))
acc_2 = sum( (predictions < 0) == (y_test < 0))/float(len(y_test))
pre = sum( (predictions >= 0) & (y_test >= 0))/float(sum(predictions >= 0))
rec = sum( (predictions >= 0) & (y_test >= 0))/float(sum(y_test >= 0))
f_score = 2 * pre * rec / (pre + rec)


print "mae: ", mae
print "cor: ", cor
print "ccc: ", ccc
print "pre: ", pre
print "rec: ", rec
print "f_score: ", f_score
print "acc_7: ", acc_7
#print "acc_5: ", acc_5
print "acc_2: ", acc_2

