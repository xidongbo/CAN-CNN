# encoding:utf-8
from sklearn.externals import joblib
import math
from gensim.models import KeyedVectors
import time
import os
import string
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from tensorflow import set_random_seed
import numpy as np

np.random.seed(10)

set_random_seed(10)


class DataHelp(object):
    def __init__(
            self,
            lower=False,
            dataname1='error',
            dataname2='error',
            dataname3='error'):
        self.lower = lower
        self.dataname1 = dataname1
        self.dataname2 = dataname2
        self.dataname3 = dataname3

    def embedding(
            self,
            texts,
            labels,
            nb_words=0,
            embedding_size=300,
            maxlens=0,
            seed=0):
        textsa, textsb, textsc = texts
        labelsa, labelsb, labelsc = labels
        texts = []
        texts.extend(textsa)
        texts.extend(textsb)
        texts.extend(textsc)
        labels = np.concatenate([labelsa, labelsb, labelsc], axis=0)
        assert len(texts) == labels.shape[0]
        nb_a, nb_b, nb_c = len(textsa), len(textsb), len(textsc)
        print('nb_samples:{},{},{}'.format(nb_a, nb_b, nb_c))
        print("lower=" + str(self.lower))
        # !"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n
        tokenizer = Tokenizer(
            num_words=nb_words,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r',
            lower=self.lower,
            char_level=False)
        char_tokenizer = Tokenizer(
            num_words=nb_words,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r',
            lower=self.lower,
            char_level=True)

        tokenizer.fit_on_texts(texts)
        char_tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        char_sequences = char_tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index
        char_index = char_tokenizer.word_index
        joblib.dump(word_index, 'word_index.pkl', compress=3)
        joblib.dump(char_index, 'char_index.pkl', compress=3)

        print('Found %d unique char.' % len(char_index))
        print('Found %d unique word.' % len(word_index))
        # get pretrained embedding matrix
        # word index start from one, zero is reserved
        # bulid word2vec model
        print('load word2vec model...')
        start = time.clock()
        word_vectors = KeyedVectors.load_word2vec_format(
            r'data/GoogleNews-vectors-negative300.bin', binary=True)
        end = time.clock()
        print('load successfully,cost %.2f s' % (end - start))
        print('number of pre-trained unique words:', len(word_vectors.vocab))
        print('word vector dim=%d' % word_vectors.vector_size)

        count = 0
        num_words = min(nb_words, len(word_index))
        if nb_words == 0:
            num_words = len(word_index)
        extend_query = 305
        # glorot uniform
        limit = np.sqrt(6. / (num_words + extend_query + embedding_size))
        embedding_matrix = np.random.uniform(-limit,
                                             limit,
                                             (num_words + extend_query,
                                              embedding_size))
        print('embedding_matrix_shape: {}'.format(embedding_matrix.shape))
        for word, i in word_index.items():
            if i >= num_words:
                continue
            try:
                embedding_vector = word_vectors[word.decode('utf8')]
                # words not found in embedding index will be random initial
                count += 1
                embedding_matrix[i] = embedding_vector
            except BaseException:
                continue
        joblib.dump(embedding_matrix, 'embedding_matrix.pkl', compress=3)

        print(
            "Found %s%% word vectors in the pretrained dict" %
            (count * 100.0 / num_words))

        lens = sorted([len(sequence) for sequence in sequences])
        print('word minlen:%d,maxlen:%d,meanlen:%d' %
              (lens[0], lens[-1], np.mean(lens)))

        char_lens = [len(sequence) for sequence in char_sequences]
        char_lens.sort()
        print('char minlen:%d,maxlen:%d,meanlen:%d' %
              (char_lens[0], char_lens[-1], np.mean(char_lens)))

        # free memory
        texts = []
        print("pad...")
        # default maxlen is the length of the longest sequence
        if maxlens == 0:
            maxlen = lens[-1]
            char_maxlen = char_lens[-1]
        else:
            char_maxlen = maxlens[0]
            maxlen = maxlens[1]
        print('word maxlen:{}'.format(maxlen))
        print('char maxlen:{}'.format(char_maxlen))

        params = {'num_words': num_words + 1,
                  'num_chars': len(char_index) + 1,
                  'num_category': labels.shape[-1],
                  'maxlen_words': maxlen,
                  'maxlen_chars': char_maxlen}
        joblib.dump(params, 'params.pkl', compress=3)
        print('params' + str(params))
        data = pad_sequences(sequences, maxlen=maxlen)
        ###########################
        data1_wordsindex = list(np.unique(data[:nb_a]))
        data2_wordsindex = list(np.unique(data[nb_a:nb_a + nb_b]))
        data3_wordsindex = list(
            np.unique(data[nb_a + nb_b:nb_a + nb_b + nb_c]))
        assert nb_a + nb_b + nb_c == np.shape(data)[0]
        joblib.dump(
            data1_wordsindex,
            'words_index_{}.pkl'.format(
                self.dataname1),
            compress=3)
        joblib.dump(
            data2_wordsindex,
            'words_index_{}.pkl'.format(
                self.dataname2),
            compress=3)
        joblib.dump(
            data3_wordsindex,
            'words_index_{}.pkl'.format(
                self.dataname3),
            compress=3)
        ##############################

        char_data = pad_sequences(char_sequences, maxlen=char_maxlen)
        print('Shape of all data tensor:', data.shape)
        print('Shape of all char_data tensor:', char_data.shape)
        data1 = data[:nb_a]
        data2 = data[nb_a:nb_a + nb_b]
        data3 = data[nb_a + nb_b:nb_a + nb_b + nb_c]
        assert data2.shape[0] == nb_b
        assert data3.shape[0] == nb_c
        char_data1 = char_data[:nb_a]
        char_data2 = char_data[nb_a:nb_a + nb_b]
        char_data3 = char_data[nb_a + nb_b:nb_a + nb_b + nb_c]
        assert char_data2.shape[0] == nb_b
        assert char_data3.shape[0] == nb_c
        labelsa = labels[:nb_a]
        labelsb = labels[nb_a:nb_a + nb_b]
        labelsc = labels[nb_a + nb_b:nb_a + nb_b + nb_c]
        assert labelsb.shape[0] == nb_b
        assert labelsc.shape[0] == nb_c

        if seed > 0:
            np.random.seed(seed)
        # no split train and test,CV
        for data, char_data, labels, dataname in zip(
            [
                data1, data2, data3], [
                char_data1, char_data2, char_data3], [
                labelsa, labelsb, labelsc], [
                    self.dataname1, self.dataname2, self.dataname3]):
            shuffle_indices = np.random.permutation(np.arange(len(labels)))
            data = data[shuffle_indices]
            char_data = char_data[shuffle_indices]
            labels = labels[shuffle_indices]
            # [char,word,labels]
            joblib.dump([char_data, data, labels],
                        '{}.pkl'.format(dataname), compress=3)
            print('{} total {} samples'.format(dataname, len(labels)))

    def MR_loader(self, url=r"data/MR/"):
        print('current basepath:' + os.path.abspath('.'))
        print('load MR dataset...')
        neg = os.path.join(url, "rt-polarity.neg")
        pos = os.path.join(url, "rt-polarity.pos")
        with open(neg, 'r') as fr:
            neg_texts = fr.readlines()
        with open(pos, 'r') as fr:
            pos_texts = fr.readlines()
        labels = np.concatenate(
            (np.array([[1, 0]] * len(neg_texts)), np.array([[0, 1]] * len(pos_texts))))
        neg_texts.extend(pos_texts)
        texts = neg_texts
        print(
            'positive:{},negative:{}'.format(
                len(neg_texts) -
                len(pos_texts),
                len(pos_texts)))
        print('nb_samlpes:{}'.format(len(labels)))
        return texts, labels

    def CR_loader(self, url=r"data/CR/"):
        print('current basepath:' + os.path.abspath('.'))
        print('load CR dataset...')
        texts = []
        labels = []
        for file in os.listdir(url):
            with open(os.path.join(url, file), 'r') as fr:
                lines = fr.readlines()
            for line in lines:
                score = 0.
                if ('##' in line) and ('[' in line):
                    l = line.split('[')
                    flag = True
                    for j in range(1, len(l)):
                        if l[j][0] in ['+', '-']:
                            if l[j][0] == l[1][0]:
                                if l[j][1] == ']':
                                    score += 1 if l[j][0] == '+' else -1
                                else:
                                    score += int(l[j][:2])
                            else:
                                flag = False
                                break
                    if flag:
                        texts.append(l[-1].split('##')[-1])
                        labels.append(1 if score > 0 else 0)
        print(
            'positive:{},negative:{}'.format(
                sum(labels),
                len(labels) -
                sum(labels)))
        print('nb_samlpes:{}'.format(len(labels)))
        labels = to_categorical(labels)
        return texts, labels

    def AFF_loader(self, url=r"data/AFF/"):
        print('current basepath:' + os.path.abspath('.'))
        print('load AFF dataset...')
        food = os.path.join(url, "foods.txt")
        neg_texts = []
        pos_texts = []
        num_each_class = 3600
        neg_count = 0
        pos_count = 0
        flag = 0
        with open(food, 'r') as fr:
            for line in fr.readlines():
                if 'review/score: 1.0' in line:
                    flag = -1
                if 'review/score: 5.0' in line:
                    flag = 1
                if 'review/text:' in line:
                    text = line[line.find(':') + 2:]
                    if flag == 1 and pos_count < num_each_class:
                        if 5 < len(text.split(' ')) < 40:
                            pos_count += 1
                            pos_texts.append(text)
                    elif flag == -1 and neg_count < num_each_class:
                        if 5 < len(text.split(' ')) < 40:
                            neg_count += 1
                            neg_texts.append(text)
                    flag = 0
        labels = np.concatenate(
            (np.array([[1, 0]] * len(neg_texts)), np.array([[0, 1]] * len(pos_texts))))
        neg_texts.extend(pos_texts)
        texts = neg_texts
        print(
            'positive:{},negative:{}'.format(
                len(neg_texts) -
                len(pos_texts),
                len(pos_texts)))
        print('nb_samlpes:{}'.format(len(labels)))
        assert labels.shape[0] == len(neg_texts)
        return texts, labels


if __name__ == '__main__':
    datahelp = DataHelp(dataname1='MR', dataname2='CR', dataname3='AFF')
    texts = []
    labels = []
    textsa, labelsa = datahelp.MR_loader()
    texts.append(textsa)
    labels.append(labelsa)
    textsb, labelsb = datahelp.CR_loader()
    texts.append(textsb)
    labels.append(labelsb)
    textsc, labelsc = datahelp.AFF_loader()
    texts.append(textsc)
    labels.append(labelsc)
    datahelp.embedding(texts=texts, labels=labels, maxlens=[250, 50])
