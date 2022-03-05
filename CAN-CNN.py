# attention not share
from keras.utils.vis_utils import plot_model as plot
from keras.engine import InputSpec
import keras.backend.tensorflow_backend as k
import tensorflow as tf
import sys
import os
import numpy as np
from keras import initializers
from keras.constraints import max_norm
from sklearn.externals import joblib
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import adam, adadelta
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Conv1D, GlobalMaxPool1D, merge, \
    Dropout, TimeDistributed, RepeatVector, Permute, LSTM, Activation, Flatten, Lambda, Layer, Reshape, BatchNormalization
from tensorflow import set_random_seed
from numpy.random import seed
seed(10)
set_random_seed(10)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
args = {
    '-gpus': '',
    '-dataset1': '',
    '-dataset2': '',
    '-model_name': '',
    '-fold': '',
    '-alpha': '',
    '-beta': '',
    '-gamma': '',
    '-norm': '',
    '-kmax': '',
    '-topk': '',
    '-topn': ''}
for i in range((len(sys.argv) - 1) / 2):
    if sys.argv[i * 2 + 1] in args:
        args[sys.argv[i * 2 + 1]] = sys.argv[i * 2 + 2]
    else:
        print('args error!')
        exit()
for i in [1]:
    if len(args['-gpus']) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = args['-gpus']
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if len(args['-dataset1']) > 0:
        dataset1 = args['-dataset1']
    else:
        print('dataset1 null!!!')
        exit()
    if len(args['-dataset2']) > 0:
        dataset2 = args['-dataset2']
    else:
        print('dataset2 null!!!')
        exit()
    if len(args['-model_name']) > 0:
        model_name = args['-model_name']
    else:
        print('model_name null!!!')
        exit()
    if len(args['-fold']) > 0:
        # 0-9
        fold = int(args['-fold'])
    if len(args['-alpha']) > 0:
        alpha = float(args['-alpha'])
    else:
        print('alpha null!!!')
        exit()
    if len(args['-gamma']) > 0:
        gamma = float(args['-gamma'])
    else:
        print('gamma null!!!')
        exit()
    if len(args['-beta']) > 0:
        beta = float(args['-beta'])
    else:
        print('beta null!!!')
        exit()
    if len(args['-norm']) > 0:
        norm = float(args['-norm'])
    else:
        norm = 1
    if len(args['-kmax']) > 0:
        kmax = int(args['-kmax'])
    else:
        kmax = 2
    if len(args['-topk']) > 0:
        topk = int(args['-topk'])
    else:
        topk = 4
    if len(args['-topn']) > 0:
        topn = int(args['-topn'])
    else:
        topn = 50
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.InteractiveSession(
    config=tf.ConfigProto(
        gpu_options=gpu_options))
k.set_session(sess)


def class_attention(train_data, train_label, k=2, oov=1):
    # train_data:numpy array
    # train_label:one-hot encoder
    # max k: y_i/\pi_{j!=i}y_j
    class_index = np.argmax(train_label, axis=1)
    num_class = train_label.shape[1]
    assert len(class_index) == len(train_data)
    class_text = dict()
    for index, item in enumerate(class_index):
        if item in class_text:
            class_text[item].append(train_data[index])
        else:
            class_text[item] = []
            class_text[item].append(train_data[index])
    assert num_class == len(class_text)
    max = 0
    class_word = []
    for i in range(num_class):
        word, times = np.unique(class_text[i], return_counts=True)
        # index is word,value is times,word zero is oov.
        word_times_array = np.zeros(word[-1] + 1)
        word_times_array[word] = times
        word_times_array += float(oov)
        class_word.append(word_times_array)
        if len(word_times_array) > max:
            max = len(word_times_array)
    for i in range(num_class):
        # pad to max_len
        while len(class_word[i]) < max:
            class_word[i] = np.append(class_word[i], float(oov))
    return_word = []
    for i in range(num_class):
        temp = class_word[i]
        for j in range(num_class):
            if j != i:
                temp = temp / class_word[j]
        # select max k word
        #return_word.extend(list(np.argpartition(temp, -k)[-k:]))
        return_word.extend(list(np.argsort(temp)[-k:]))
    assert len(return_word) == k * num_class
    return return_word


filter_sizes = [3, 4, 5]
embedded_dim = 300
num_filters = 100
#dropout = 0.5
batch_size = 128
epsilon1 = 10e-8
_epsilon = tf.convert_to_tensor(epsilon1, tf.float32)
###########################
lr = 1e-3
###########################
earlystop = 3
num_epoches = 100000
basepath = "preprocessed_3data/"
params = joblib.load(os.path.join(basepath, "params.pkl"))
lookup_table = joblib.load(
    os.path.join(
        basepath,
        "embedding_matrix.pkl"))
alldata1 = joblib.load(os.path.join(basepath, "{}.pkl".format(dataset1)))
alldata2 = joblib.load(os.path.join(basepath, "{}.pkl".format(dataset2)))
print('dataset1: {}'.format(dataset1))
print('dataset2: {}'.format(dataset2))
print('params: {}'.format(str(params)))
print(lookup_table.shape)
# prepare data
alltrain, alltrain_label = [], []
alldev, alldev_label = [], []
alltest, alltest_label = [], []

for alldata in [alldata1, alldata2]:
    data = alldata[1]
    labels = alldata[2]
    num_val = len(labels) / 10
    if fold == 9:
        right = len(labels)
    else:
        right = (fold + 1) * num_val
    test = data[fold * num_val:right]
    test_label = labels[fold * num_val:right]
    train = np.delete(data, np.s_[fold * num_val:right], axis=0)
    train_label = np.delete(labels, np.s_[fold * num_val:right], axis=0)
    index = int(0.9 * len(train_label))
    dev = train[index:]
    dev_label = train_label[index:]

    train = train[:index]
    train_label = train_label[:index]

    alltrain.append(train)
    alltrain_label.append(train_label)
    alldev.append(dev)
    alldev_label.append(dev_label)
    alltest.append(test)
    alltest_label.append(test_label)

print('data1, train:{}, dev:{}, test:{}'.format(
    len(alltrain[0]), len(alldev[0]), len(alltest[0])))
print('data2, train:{}, dev:{}, test:{}'.format(
    len(alltrain[1]), len(alldev[1]), len(alltest[1])))

words = class_attention(alltrain[0], alltrain_label[0], k=topn, oov=1)
word_index = joblib.load(
    os.path.join(
        basepath,
        "word_index.pkl"))
index_word = dict(zip(word_index.values(), word_index.keys()))

for i in range(params['num_category']):
    for j in range(topn):
        print('class {}, index: {}, word: {}'.format(
            i, words[i * topn + j], index_word[words[i * topn + j]]))
# neg_words=[]
# neg_count=0
# for neg in ['no','No','not','Not','don\'t','Don\'t','didn\'t','dont','Dont','didnt']:
#     if word_index.has_key(neg):
#         neg_count+=1
#         neg_words.append(word_index[neg])
#         #print('word:{},index:{}'.format(neg,word_index[neg]))


class Attention(Layer):
    # compute general attention: weight=softmax(h^T_s tanh(Wh_t+b)),sum(weight*ht)
    # h_s is query, h_t is key
    # query=inputs[0]:(None,dim)
    # key=inputs[1]:(None,time_step,dim)
    # return (None,time_step),(None,dim)
    def __init__(
            self,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        assert len(input_shape[0]) == 2
        assert len(input_shape[1]) == 3
        # shape=(dim,dim)
        self.w = self.add_weight(name='att_weight',
                                 shape=(input_shape[1][-1],
                                        input_shape[1][-1]),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        # shape=(dim)
        self.b = self.add_weight(name='att_weight', shape=(
            input_shape[1][-1],), initializer=self.bias_initializer, trainable=True)
        self.built = True

    def call(self, inputs):
        query, key = inputs
        time_step = k.shape(key)[-2]
        dim = k.shape(key)[-1]
        # (None, time_step, dim)*(dim,dim)=(None, time_step, dim)
        wh = k.dot(key, self.w) + self.b
        wh = k.tanh(wh)
        # (None,time_step,dim)
        query = k.repeat(query, time_step)
        # (None, time_step)
        a = k.sum(query * wh, axis=-1)
        return_weight = a
        weight = k.softmax(a)

        # (None, dim,time_step)
        a = k.repeat(weight, dim)
        # (None,time_step,dim)
        a = k.permute_dimensions(a, (0, 2, 1))
        # (None,dim)
        weighted_vec = k.sum(a * key, axis=-2, keepdims=False)
        return [weight, weighted_vec, return_weight]

    def compute_output_shape(self, input_shape):
        # key shape
        input_shape = input_shape[1]
        return [(input_shape[0], input_shape[1]), (input_shape[0],
                                                   input_shape[2]), (input_shape[0], input_shape[1])]


# (batch_size,maxlen_words)
inputs1 = []
inputs2 = []
embedded_memory1 = []
embedded_memory2 = []
input_x1 = Input(shape=(params['maxlen_words'],), dtype='int32')
input_x2 = Input(shape=(params['maxlen_words'],), dtype='int32')
inputs1.append(input_x1)
inputs2.append(input_x2)
# (batch_size,maxlen_words,embedding_dim)
embedding = Embedding(input_dim=lookup_table.shape[0],
                      output_dim=embedded_dim,
                      weights=[lookup_table],
                      trainable=True)
for x in range(params['num_category']):
    if x == 0:  # neg label
        input_x1_memory = Input(shape=(topn,), dtype='int32')  # +neg_count
        input_x2_memory = Input(shape=(topn,), dtype='int32')
    else:
        input_x1_memory = Input(shape=(topn,), dtype='int32')
        input_x2_memory = Input(shape=(topn,), dtype='int32')
    inputs1.append(input_x1_memory)
    inputs2.append(input_x2_memory)

    # [(None,topn,dim),]*params['num_category']
    embedded_memory1.append(embedding(input_x1_memory))
    embedded_memory2.append(embedding(input_x2_memory))
assert len(inputs1) == len(inputs2) == params['num_category'] + 1
assert len(embedded_memory1) == len(embedded_memory2) == params['num_category']
# (None,maxlen,dim)
embedded_x1 = embedding(input_x1)
embedded_x2 = embedding(input_x2)

attention = Attention()
convs = []
assert topk <= topn
for filter_size in filter_sizes:
    # (batch_size,time_step-kernel_size+1,num_filters)
    conv = Conv1D(filters=num_filters,
                  kernel_size=filter_size,
                  strides=1,
                  padding='valid',
                  activation='relu')
    convs.append(conv)


class Select(Layer):
    def __init__(self, **kwargs):
        super(Select, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        # embedded_x:(None,time_step,dim)
        # embedded_memory:(None,topn,dim)
        # return [None,dim]*topk,(None,dim),# (None, topk)
        output = [(input_shape[0][0], input_shape[0][2])] * topk
        output.append((input_shape[0][0], input_shape[0][2]))
        output.append((input_shape[0][0], topk))
        return output

    def call(self, inputs):
        embedded_x, embedded_memory = inputs
        # (None,time_step,dim)->(None,dim) (fasttext-ave)
        #cont = k.mean(embedded_x, axis=-2, keepdims=False)
        # cont:(None,dim)
        # embedded_x:(None,time_step,dim)
        # embedded_memory:(None,topn,dim)
        # from embedded_memory select topk querys which are similar with context, return list
        # (None,topn,dim)*(None,time_step,dim)->(None,topn,time_step)
        dot = tf.matmul(embedded_memory, embedded_x, transpose_b=True)
        # (None,time_step)
        norm_a = k.sqrt(k.sum(embedded_x * embedded_x, axis=-1))
        # (None,topn)
        norm_b = k.sqrt(k.sum(embedded_memory * embedded_memory, axis=-1))
        # (None,topn,1)*(None,1,time_step)->(None, topn,time_step)
        norm_ab = tf.matmul(k.expand_dims(norm_b, -1),
                            k.expand_dims(norm_a, 1))
        sim = dot / (norm_ab)
        # (None, topn,time_step)->(None, topn,1)
        sim, _ = tf.nn.top_k(sim, k=1, sorted=True)
        # (None, topn,1)->(None, topn)
        sim = k.squeeze(sim, axis=-1)
        # (None, topn)->(None, topk)
        _, max_index = tf.nn.top_k(sim, k=topk, sorted=True)
        return_index = max_index
        #####################
        # select querys from 'embedded_memory' according to the indics 'max_index'
        ####################
        n = tf.shape(embedded_memory)[0]  # batch_size
        # (None,topk)
        ii = tf.tile(tf.range(n)[:, tf.newaxis], (1, int(max_index.shape[-1])))
        # Make tensor of indices for the first dimension
        max_index = tf.stack([ii, max_index], axis=-1)
        # (None,topk,dim)
        querys = tf.gather_nd(params=embedded_memory, indices=max_index)
        # [None,dim]*topk
        output = tf.unstack(querys, axis=1)
        # output.append(cont)
        output.append(return_index)
        return output


class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last
        # dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)


transform_dense = Dense(embedded_dim)


def attention_block(embedded_x, embedded_memory):
    attentioned = []
    class_weights = []
    return_indics = []
    for i in range(params['num_category']):
        # dynamic select topk from topn for each category
        weights = []
        selected = Select()([embedded_x, embedded_memory[i]])
        selected_querys, return_index = selected[:-1], selected[-1]
        return_indics.append(return_index)
        assert len(selected_querys) == topk
        for query_index, embedded_query in enumerate(selected_querys):
            # for count in range(1):
            # (None,time_steps),(None,dim)
            norm_weight, att, weight = attention([embedded_query, embedded_x])
            # embedded_query=merge([att,embedded_query],mode='ave')
            if query_index == 0 and i == 0:
                # neg class first query
                saved_weight = weight

            weights.append(norm_weight)
            attentioned.append(att)
        # weight distribution
        if topk > 1:
            dis_weight = merge(weights, mode='ave')
        else:
            dis_weight = weights[0]
        class_weights.append(dis_weight)
    ps, qs = [], []
    for i in range(params['num_category']):
        p = class_weights[i]
        q = class_weights[:i]
        q.extend(class_weights[i + 1:])
        if params['num_category'] > 2:
            q = merge(q, mode='ave')
        else:
            q = q[0]
        ps.append(p)
        qs.append(q)
    merged1 = merge(attentioned, mode='sum')
    # CNN
    pooled_outputs = []
    for conv in convs:
        # (batch_size,num_filters)
        pooled_outputs.append(KMaxPooling(k=kmax)(conv(embedded_x)))

    # (batch_size,num_filters)
    merged2 = merge(pooled_outputs, mode='sum')
    # merged2=transform_dense(merged2)
    merged = merge([merged1, merged2], mode='concat')
    return merged, ps, qs, merged1, merged2, return_indics, saved_weight


mergeda, psa, qsa, att_contexta, cnn_contexta, return_indics, saved_weight = attention_block(
    embedded_x1, embedded_memory1)
mergedb, psb, qsb, att_contextb, cnn_contextb, _, _ = attention_block(
    embedded_x2, embedded_memory2)
###################shared##################
densed = Dense(
    units=params['num_category'],
    activation='softmax',
    name='class',
    kernel_constraint=max_norm(norm))
output = densed(mergeda)
target_pred = densed(mergedb)


def kl(ps, qs):
    s = 0.
    for i in range(params['num_category']):
        p, q = ps[i], qs[i]
        p = tf.clip_by_value(p, _epsilon, 1. - _epsilon)
        q = tf.clip_by_value(q, _epsilon, 1. - _epsilon)
        kl = k.sum(p * k.log(p) - p * k.log(q), axis=-1, keepdims=False)
        s += kl
    return s


def sim_mse(space1, space2):
    # (dim,None)
    space_t = k.permute_dimensions(space1, (1, 0))
    # (None, 1)
    s1 = k.sqrt(k.sum(k.square(space1), axis=1, keepdims=True))
    # (1, None)
    s2 = k.permute_dimensions(s1, (1, 0))
    # (None,None)
    distance1 = k.dot(space1, space_t) / (k.dot(s1, s2))

    # (dim,None)
    space_t = k.permute_dimensions(space2, (1, 0))
    # (None, 1)
    s1 = k.sqrt(k.sum(k.square(space2), axis=1, keepdims=True))
    # (1, None)
    s2 = k.permute_dimensions(s1, (1, 0))
    # (None,None)
    distance2 = k.dot(space2, space_t) / (k.dot(s1, s2))
    dis = k.mean(k.square(distance1 - distance2), axis=-1)
    return dis


def loss1(y_true, y_pred):
    # kl divergence
    s = kl(psa, qsa) + kl(psb, qsb)
    return k.categorical_crossentropy(y_true, y_pred)\
        - alpha * s


def loss2(y_true, y_pred):
    # (None,dim)
    source = att_contexta
    target = att_contextb
    # MMD, not consider the RKHS(Reproducing Kernel Hilbert Space)
    #  (batch_size,num_filters)->(num_filters,)
    MMD = k.sum(k.square(k.mean(mergeda, axis=0) - k.mean(mergedb, axis=0)))
    # if the loss function doesn't use the input, will note that 'ValueError:
    # None values not supported'
    return 0. * k.categorical_crossentropy(y_true, y_pred)\
        + gamma * MMD\
        + beta * sim_mse(source, cnn_contexta)\
        + beta * sim_mse(target, cnn_contextb)


sgd = adam(lr=lr)
# [x1,topn,topn,,,x2,topn,topn,,,]
inputs1.extend(inputs2)
model = Model(inputs=inputs1, outputs=[output, target_pred])
# plot(model,to_file="model.png",show_shapes=True,show_layer_names=True)
model.compile(
    loss=[loss1, loss2],
    optimizer=sgd,
    metrics=['accuracy'])
monitor = 'val_class_acc_2'
mode = 'max'
early_stopping = EarlyStopping(monitor=monitor, patience=earlystop, mode=mode)
checkpoint = ModelCheckpoint(
    model_name,
    monitor=monitor,
    verbose=0,
    save_best_only=True,
    mode=mode)

num_train1 = alltrain_label[0].shape[0]
num_train2 = alltrain_label[1].shape[0]
num_dev1 = alldev_label[0].shape[0]
num_dev2 = alldev_label[1].shape[0]
num_test1 = alltest_label[0].shape[0]
num_test2 = alltest_label[1].shape[0]


def data_generator(type='train'):
    if type == 'train':
        # [data1,data2]
        data = alltrain
        labels = alltrain_label
    elif type == 'dev':
        data = alldev
        labels = alldev_label
    else:
        print('type error:{}'.format(type))
        exit()
    #num_classquery = topn * params['num_category']
    data1 = np.concatenate([data[0], labels[0]], axis=-1)
    data2 = np.concatenate([data[1], labels[1]], axis=-1)
    num_batch1 = data1.shape[0] // batch_size
    num_batch2 = data2.shape[0] // batch_size
    i, j = 0, 0
    while True:
        if i >= num_batch1:
            i = 0
            np.random.shuffle(data1)
        if j >= num_batch2:
            j = 0
            np.random.shuffle(data2)
        batch_data1 = data1[i * batch_size:(i + 1) * batch_size]
        batch_data2 = data2[j * batch_size:(j + 1) * batch_size]
        i += 1
        j += 1
        assert batch_data1.shape[1] == params['maxlen_words'] + \
            params['num_category']
        querys1, querys2 = [], []
        querys1.append(batch_data1[:, :params['maxlen_words']])
        querys2.append(batch_data2[:, :params['maxlen_words']])
        # for ii in range(params['num_category']):
        q1 = words[0:topn]
        # q1.extend(neg_words)
        q2 = list(params['num_words'] + np.array(range(topn)) + 1)
        # q2.extend(neg_words)
        querys1.append(np.repeat([q1], batch_size, axis=0))
        querys2.append(np.repeat([q2], batch_size, axis=0))

        querys1.append(
            np.repeat([words[1 * topn:1 * topn + topn]], batch_size, axis=0))
        querys2.append(np.repeat(
            [params['num_words'] + np.array(range(topn)) + topn * 1 + 1], batch_size, axis=0))
        querys1.extend(querys2)
        # X,Y
        yield querys1, [batch_data1[:, params['maxlen_words']:],
                        batch_data2[:, params['maxlen_words']:]]


######get the weight#########
layer_em = model.get_layer('embedding_1')
em_before = layer_em.get_weights()[0]
history = model.fit_generator(generator=data_generator('train'),
                              steps_per_epoch=20,
                              # min(num_train1,num_train2)//batch_size+1,
                              epochs=num_epoches,
                              verbose=1,
                              callbacks=[early_stopping, checkpoint],
                              validation_data=data_generator('dev'),
                              validation_steps=num_dev2 // batch_size)  # max(num_dev1,num_dev2)//batch_size)
# print(history.history)
#############################
# load the best model
model.load_weights(model_name)


def cos(w, embeded):
    topn = 100
    wv = embeded[w]
    embeded = embeded[:params['num_words']]
    dot = np.dot(wv, np.transpose(embeded))
    norm_wv = np.sqrt(np.sum(wv**2, axis=-1))
    norm_em = np.sqrt(np.sum(embeded**2, axis=-1))
    sim = dot / norm_wv / norm_em
    sim = list(np.argsort(sim)[-topn:])
    sim.reverse()
    return sim


######get the weight#########
layer_em = model.get_layer('embedding_1')
em_after = layer_em.get_weights()[0]
index_word[0] = None
# output the similar CMM before train and after train
# similary only count in data1 or data2
data1_wordsindex = joblib.load(
    os.path.join(
        basepath,
        "words_index_{}.pkl".format(dataset1)))
data2_wordsindex = joblib.load(
    os.path.join(
        basepath,
        "words_index_{}.pkl".format(dataset2)))
# source
# for ww in words:
#     ww1=''
#     for ww11 in cos(ww, em_before)[1:]:
#         if ww11 in data1_wordsindex:
#             ww1=ww11
#             break
#     if ww1 not in data1_wordsindex:
#         print('error1 Word:{},{}'.format(ww1,index_word[ww1]))
#         exit()
#     ww2=''
#     for ww22 in cos(ww, em_after)[1:]:
#         if ww22 in data1_wordsindex:
#             ww2=ww22
#             break
#     if ww2 not in data1_wordsindex:
#         print('error2 Word:{},{}'.format(ww2,index_word[ww2]))
#         exit()
#     print('Word:{},{}, simlary:{},{}'.format(ww,index_word[ww],index_word[ww1],index_word[ww2]))
# target CMM
# mm=0
# for ii in range(params['num_category']*topn):
#     ww=params['num_words'] + ii+1
#     ww1 = ''
#     ww1_list=[]
#     cc=0
#     for ww11 in cos(ww, em_before):#[1:]
#         if ww11 in data2_wordsindex:
#             ww1 = ww11
#             ww1_list.append(ww1)
#             cc+=1
#             if cc>0:
#                 break
#     ww2_list=[]
#     cc=0
#     ww2 = ''
#     for ww22 in cos(ww, em_after):#[1:]
#         if ww22 in data2_wordsindex:
#             ww2 = ww22
#             ww2_list.append(ww2)
#             cc+=1
#             if cc>6:
#                 break
#     mm+=1
#     if mm==topn+1:
#         print('##############################################')
#     print('Word:{},{}, simlary:{};{}'.format(ww, None, [index_word[ww1] for ww1 in ww1_list], [index_word[ww2] for ww2 in ww2_list]))
# exit()
#############################
alltest[0] = [alltest[0]]
q1 = words[0 * topn:0 * topn + topn]
# q1.extend(neg_words)
alltest[0].append(np.repeat([q1], num_test1, axis=0))
alltest[0].append(
    np.repeat([words[1 * topn:1 * topn + topn]], num_test1, axis=0))
alltest[0].extend(alltest[0])

alltest[1] = [alltest[1]]
q2 = list(params['num_words'] + np.array(range(topn)) + topn * 0 + 1)
# q2.extend(neg_words)
alltest[1].append(np.repeat([q2], num_test2, axis=0))
alltest[1].append(np.repeat([params['num_words'] +
                  np.array(range(topn)) + topn * 1 + 1], num_test2, axis=0))
alltest[1].extend(alltest[1])
# test data1
eva_data1 = model.evaluate(
    x=alltest[0], y=[
        alltest_label[0], alltest_label[0]], batch_size=batch_size, verbose=2)
print(eva_data1)
# test data2
eva_data2 = model.evaluate(
    x=alltest[1], y=[
        alltest_label[1], alltest_label[1]], batch_size=batch_size, verbose=2)
print(eva_data2)
# output=return_indics
# output.append(saved_weight)
# modeltest = Model(inputs=inputs1, outputs=output)
# pred=modeltest.predict(alltest[0])

# print(alltest[0][0][0])    #x
# print(alltest[0][1][0])    #neg
# print(alltest[0][2][0])    #pos
'''
for neg,pos,weight,sample in zip(pred[0],pred[1],pred[2],alltest[0][0]):
    sent=''
    for ww in sample:
        if ww==0:
            continue
        sent=sent+index_word[ww]+' '
    print(sent)
    print(weight)
    for w in neg:
        #print(w)
        #print(words[w+i*topn])
        if w>=topn:
            print(index_word[neg_words[w-topn]])
        else:
            print(index_word[words[w]])
    for w in pos:
        #print(w)
        #print(words[w+i*topn])
        print(index_word[words[w +topn]])
'''
