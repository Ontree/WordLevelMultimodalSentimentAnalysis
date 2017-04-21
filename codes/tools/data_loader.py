import os
from collections import defaultdict
import numpy as np
import random
import scipy.io as sio
import cPickle
import h5py


data_path = '/usr0/home/minghai1/multimodal/data/'
dataset_path = data_path + 'mosi/'

truth_path = dataset_path + 'Meta_data/boundaries_sentimentint_avg.csv'

#openface_path = dataset_path + "Features/Visual/OPEN_FACE/Segmented/"
openface_path = dataset_path + "Features/Visual/OpenfaceRaw/"
facet_path = dataset_path + "Features/Visual/FACET_GIOTA/"
covarep_path = dataset_path + "Features/Audio/raw/"
transcript_path = dataset_path + 'Transcript/SEGMENT_ALIGNED/'

word2ix_path = data_path + 'glove_word_embedding/word2ix_300_mosi.pkl'
word_embedding_path = data_path + "glove_word_embedding/glove_300_mosi.pkl"

def load_word_embedding():
    with open(word_embedding_path) as f:
        return cPickle.load(f)

'''
def load_word_embedding(word2ix, embedding_vecor_length):
    word_embedding_path = word_embedding_path_prefix + str(embedding_vecor_length) + "d.txt"
    print "## Loading word embedding: ", word_embedding_path
    with open(word_embedding_path) as f:
        lines = f.read().split("\n")
    word_num = len(word2ix.keys())
    word_embedding = [[0] * embedding_vecor_length for x in range(word_num + 1)]    #[[0] * embedding_vecor_length]
    for l in lines:
        l = l.split(' ')
        word = l[0].upper()
        if (word in word2ix) and word2ix[word] != 0:
            word_embedding[word2ix[word]] = [float(x) for x in l[1:]]
    return np.array(word_embedding)
'''

def load_word2ix():
    with open(word2ix_path) as f:
        word2ix = cPickle.load(f)
    return word2ix


# load meta data truth_dict[video_id][seg_id]
def load_truth():
    truth_dict = defaultdict(dict)
    with open(truth_path) as f:
        lines = f.read().split("\r\n")
    for line in lines:
        if line != '':
            line = line.split(",")
            truth_dict[line[2]][line[3]] = {'start_time': float(line[0]), 'end_time':float(line[1]), 'sentiment':float(line[4])}
    return truth_dict


def load_facet(truth_dict):
    for video_index in truth_dict:
        file_name = facet_path + video_index + '.FACET_out.csv'
        #print file_name
        with open(file_name) as f:
            lines = f.read().split('\r\n')[1:]
            lines = [[float(x) for x in line.split(',')]  for line in lines if line != '']
            for seg_index in truth_dict[video_index]:
                for w in truth_dict[video_index][seg_index]['data']:
                    start_frame = int(w['start_time_clip']*30)
                    end_frame = int(w['end_time_clip']*30)
                    ft = [line[5:] for line in lines[start_frame:end_frame]]
                    if ft == []:
                        avg_ft =  np.zeros(len(lines[0]) - 5)
                    else:
                        #print np.array(ft).shape
                        #print ft[0]
                        avg_ft = np.mean(ft,0)
                    w['facet'] = avg_ft


def load_covarep(truth_dict):
    for video_index in truth_dict:
        file_name = covarep_path + video_index + '.mat'
        fts = sio.loadmat(file_name)['features']
        #print fts.shape
        for seg_index in truth_dict[video_index]:
            for w in truth_dict[video_index][seg_index]['data']:
                start_frame = int(w['start_time_clip']*100)
                end_frame = int(w['end_time_clip']*100)
                ft = fts[start_frame:end_frame]
                if ft.shape[0] == 0:
                    avg_ft = np.zeros(ft.shape[1])
                else:
                    #print np.array(ft).shape
                    #print ft[0]
                    avg_ft = np.mean(ft,0)
                avg_ft[np.isnan(avg_ft)] = 0
                avg_ft[np.isneginf(avg_ft)] = 0
                w['covarep'] = avg_ft


def load_transcript(truth_dict, word2ix):
    for video_index in truth_dict:
        for seg_index in truth_dict[video_index]:
            file_name = transcript_path + video_index + '_' + seg_index
            truth_dict[video_index][seg_index]['data'] = []
            with open(file_name) as f:
                lines = f.read().split("\n")
                for line in lines:
                    if line == '':
                        continue
                    line = line.split(',')
                    #print line
                    truth_dict[video_index][seg_index]['data'].append({'word_ix': word2ix[line[1]], 'word': line[1], 'start_time_seg': float(line[2]), 'end_time_seg':float(line[3]), 'start_time_clip':float(line[4]), 'end_time_clip':float(line[5])})


def split_data(tr_proportion, truth_dict):
    data = [(vid, truth_dict[vid]) for vid in truth_dict]
    data.sort(key = lambda x: x[0])
    tr_split = int(round(len(data) * tr_proportion))
    train = data[:tr_split]
    test = data[tr_split:]
    return train, test


def get_data(dataset, max_segment_len):
    data = {'facet': [], 'covarep': [], 'text': [], 'label': [], 'id': []}
    for i in range(len(dataset)):
        #print dataset[i][0]
        v = dataset[i][1]
        for seg_id in v:
            fts = v[seg_id]['data']
            facet, text, covarep = [], [], []
            for w in fts[:max_segment_len]:
                text.append(w['word_ix']) 
                covarep.append(w['covarep'])
                facet.append(w['facet'])
            for j in range(max_segment_len-len(text)):
                text.append(0)
                covarep.append(np.zeros(len(covarep[0])))
                facet.append(np.zeros(len(facet[0])))
            data['facet'].append(facet)
            data['covarep'].append(covarep)
            data['text'].append(text)
            data['label'].append(v[seg_id]['sentiment'])
            data['id'].append(dataset[i][0]+'_'+seg_id)
    data['facet'] = np.array(data['facet'])
    data['covarep'] = np.array(data['covarep'])
    data['text'] = np.array(data['text'])
    data['label'] = np.array(data['label'])
    return data



def load_word_level_features(max_segment_len, tr_proportion):
    word2ix = load_word2ix()
    truth_dict = load_truth()
    load_transcript(truth_dict, word2ix)
    load_facet(truth_dict)
    load_covarep(truth_dict)
    train, test = split_data(tr_proportion, truth_dict)
    train = get_data(train, max_segment_len)
    test = get_data(test, max_segment_len)
    return train, test





