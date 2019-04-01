import codecs
import numpy as np
import tensorflow as tf
from configs import bcolors
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.objectives import categorical_crossentropy
from keras.layers import Conv2D, MaxPooling2D, concatenate, Dense, Activation, Input, Reshape
from Model import WordPreprocessor, WordVecModeler
from sklearn import preprocessing

class Trainer1:
    def __init__(self, word_dim, assign_size, TB_SUMMARY_DIR, filter=128, max_word_size=500, stack=2):
        self.word_dim = int(word_dim)
        self.assign_size = assign_size
        self.TB_SUMMARY_DIR = TB_SUMMARY_DIR
        self.max_word_size = int(max_word_size)
        self.stack = stack
        self.input = Input(shape = (self.max_word_size, self.word_dim), name='input')
        using_x = Reshape((self.max_word_size, self.word_dim, 1), name='reshape_using')(self.input)

        x1 = Conv2D(filter, kernel_size=[2, word_dim], activation='relu', name='block1_conv1')(using_x)
        x1 = MaxPooling2D(pool_size = [max_word_size - 1, 1], strides = 1, name='block1_pool1')(x1)
        x1 = Reshape((1, filter), name='reshape_x1')(x1)
        self.r_result1 = Reshape(target_shape=(filter,), name='reshape_x1_filter')(x1)

        x2 = Conv2D(filter, kernel_size=[3, word_dim], activation='relu', name='block2_conv2')(using_x)
        x2 = MaxPooling2D(pool_size = [max_word_size - 2, 1], strides = 1, name='block2_pool2')(x2)
        x2 = Reshape((1, filter), name='reshape_x2')(x2)
        self.r_result2 = Reshape(target_shape=(filter,), name='reshape_x2_filter')(x2)

        x3 = Conv2D(filter, kernel_size=[4, word_dim], activation='relu', name='block3_conv3')(using_x)
        x3 = MaxPooling2D(pool_size = [max_word_size - 3, 1], strides = 1, name='block3_pool3')(x3)
        x3 = Reshape((1, filter), name='reshape_x3')(x3)
        self.r_result3 = Reshape(target_shape=(filter,), name='reshape_x3_filter')(x3)

        self.result = concatenate([self.r_result1, self.r_result2, self.r_result3], axis = 1, name='concatenate')
        self.result = Dense(units = assign_size, name='before_softmax')(self.result)
        self.result = Activation('softmax', name='predictions')(self.result)

        self.model = Model(inputs = self.input, outputs = self.result)


    def train_nonlearning(self, train_x, train_y, test_x, test_y, epoch=5, batch_size=5):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.45

        adam = optimizers.Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model.fit(x = train_x, y = train_y, validation_data = (test_x, test_y), batch_size = batch_size, epochs = epoch)
        self.model.save_weights('./Model1.h5')
        score = self.model.evaluate(test_x, test_y, verbose=0)
        print('\n')
        print('Overall Test score: ', score[0])
        print('Overall Test accuracy: ', score[1])

    def train_learning(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.45

        self.model.load_weights('./Model1.h5')
        print(bcolors.OKBLUE + 'Model1 loaded' + bcolors.ENDC)


if __name__ == '__main__':
    word_dim = 300
    test_percent = 0.1
    sum_vec = 1
    word_preprocessor = WordPreprocessor()
    word_vec_modeler = WordVecModeler(dim=word_dim)
    raw_x = word_preprocessor.data_preprocessing('input/desc.txt')
    assign_file = codecs.open('input/assignTo.txt', "r", "utf-8")
    max_word = 500

    TB_SUMMARY_DIR = './onlycnn/add1/filter512/234/relu'

    raw_y = []
    for line in assign_file:
        line = line.strip()
        raw_y.append(line)
    # load data end
    le = preprocessing.LabelEncoder()
    enc = preprocessing.OneHotEncoder()
    le.fit(raw_y)
    assign_num = len(set(raw_y))
    y_to_number = np.array(le.transform(raw_y))
    y_to_number = np.reshape(y_to_number, [-1, 1])
    enc.fit(y_to_number)
    one_hot_y = enc.transform(y_to_number).toarray()
    # assign one_hot incoding

    word_vec_modeler.load_word_vec("GoogleNews-vectors-negative300.bin")
    vec_x = None
    vec_size = []
    for idx, doc in enumerate(raw_x):
        temp_arr = []
        for word_idx, word in enumerate(doc):
            if word_vec_modeler.get_vector_from_word(word) is not None:
                temp_arr.append(word_vec_modeler.get_vector_from_word(word))
            if word_idx == max_word - 1:
                break
        if len(temp_arr) == 0:
            temp_arr.append(np.zeros(300))
        if len(temp_arr) < max_word:
            for i in range(max_word - len(temp_arr)):
                temp_arr.append(np.zeros(word_dim).tolist())
        if vec_x is None:
            vec_x = np.array(temp_arr)
        else:
            vec_x = np.vstack((vec_x, temp_arr))
        vec_size.append(len(temp_arr))
    vec_x = np.reshape(vec_x, (-1, max_word, word_dim))

    resize_vec = np.zeros((len(vec_x), max_word // sum_vec, word_dim))
    for idx, doc in enumerate(vec_x):
        temp_doc_vec = np.zeros((max_word // sum_vec, word_dim))
        for i in range(max_word // sum_vec):
            temp_vec = np.zeros(word_dim)
            for j in range(sum_vec):
                temp_vec = temp_vec + doc[i * sum_vec + j]
            temp_doc_vec[i] = temp_vec
        resize_vec[idx] = temp_doc_vec

    train_vec_x = resize_vec[:int(len(vec_x) * (1 - test_percent))]
    test_vec_x = resize_vec[int(len(vec_x) * (1 - test_percent)):]
    train_one_hot_y = one_hot_y[:int(len(one_hot_y) * (1 - test_percent))]
    test_one_hot_y = one_hot_y[int(len(one_hot_y) * (1 - test_percent)):]

    hs_trainer = Trainer1(assign_size=assign_num, word_dim=word_dim, max_word_size=500 / sum_vec,
                         TB_SUMMARY_DIR=TB_SUMMARY_DIR, filter=512)

    hs_trainer.train_nonlearning(train_x=train_vec_x, train_y=train_one_hot_y, test_x=test_vec_x,
                        test_y=test_one_hot_y, epoch=50, batch_size=4)