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



def visual_controller(facet_dim):
    #facet_dim = facet_input.shape[1]
	out_dim = 2

	model = Sequential()

	model.add(Dense(32, bias_initializer='zeros',activation='sigmoid', input_shape=(facet_dim,)))
	#model.add(Dense(out_dim, activation='sigmoid'))
        model.add(Dense(out_dim, activation='softmax'))
	return model


def my_loss(realy, predy):
	# assume realy is R_k, which is mae from trained lstm model
	# then predy is p(v | params)
        return tf.reduce_sum(tf.log(predy+1e-8) * realy, 1)


if __name__ == '__main__':
	n = 50
	d = 10	
	facet_input = np.ones((n, d))

	model = visual_controller(facet_input)

	model.compile(loss=my_loss,
				  optimizer='adam',
				  metrics=['accuracy'])


	train_set_y = np.ones((n, 1))
	model.fit(facet_input, train_set_y, 
			  batch_size=32, epochs=10, verbose=1)
	
