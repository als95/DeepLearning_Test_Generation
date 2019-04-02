import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.objectives import categorical_crossentropy
from keras.layers import Conv2D, MaxPooling2D, concatenate, Dense, Activation, Input, Reshape

from w2v_model import WordPreprocessor, WordVecModeler
from sklearn import preprocessing
import codecs

class TriageNetwork:

    def __init__(self):
        self.word_dim = 300
        self.max_word_size = 500
        self.stack = 2
        self.sum_vec = 1
        self.filter = 128
        self.word_preprocessor = WordPreprocessor()
        self.raw_data = self.word_preprocessor.data_preprocessing('input/desc.txt')
        self.word_vec_modeler = WordVecModeler(dim=self.word_dim)
        self.word_vec_modeler.load_word_vec("GoogleNews-vectors-negative300.bin")
        self.assign_size = 0

    def load_data(self):
        vec = None
        vec_size = []
        for idx, doc in enumerate(self.raw_data):
            temp_arr = []
            for word_idx, word in enumerate(doc):
                if self.word_vec_modeler.get_vector_from_word(word) is not None:
                    temp_arr.append(self.word_vec_modeler.get_vector_from_word(word))
                if word_idx == self.max_word_size - 1:
                    break

            if len(temp_arr) == 0:
                temp_arr.append(np.zeros(self.word_dim))

            if len(temp_arr) < self.max_word_size:
                for _ in range(self.max_word_size - len(temp_arr)):
                    temp_arr.append(np.zeros(self.word_dim).tolist())

            if vec is None:
                vec = np.array(temp_arr)
            else:
                vec = np.vstack((vec, temp_arr))

            vec_size.append(len(temp_arr))
        vec = np.reshape(vec, (-1, self.max_word_size, self.word_dim))

        resize_vec = np.zeros((len(vec), self.max_word_size // self.sum_vec, self.word_dim))
        for idx, doc in enumerate(vec):
            temp_doc_vec = np.zeros((self.max_word_size // self.sum_vec, self.word_dim))
            for i in range(self.max_word_size // self.sum_vec):
                temp_vec = np.zeros(self.word_dim)
                for j in range(self.sum_vec):
                    temp_vec = temp_vec + doc[i * self.sum_vec + j]
                temp_doc_vec[i] = temp_vec
            resize_vec[idx] = temp_doc_vec

        test_percent = 0.1
        assign_file = codecs.open('input/assignTo.txt', "r", "utf-8")
        raw_label = []

        for line in assign_file:
            line = line.strip()
            raw_label.append(line)

        le = preprocessing.LabelEncoder()
        enc = preprocessing.OneHotEncoder()
        le.fit(raw_label)
        self.assign_size = len(set(raw_label))
        label_to_number = np.array(le.transform(raw_label))
        label_to_number = np.reshape(label_to_number, [-1, 1])
        enc.fit(label_to_number)
        labels = enc.transform(label_to_number).toarray()

        train_datas = resize_vec[:int(len(vec) * (1 - test_percent))]
        test_datas = resize_vec[int(len(vec) * (1 - test_percent)):]
        train_labels = labels[:int(len(labels) * (1 - test_percent))]
        test_labels = labels[int(len(labels) * (1 - test_percent)):]

        return (train_datas, train_labels), (test_datas, test_labels)

    def load_model(self, name_of_file):
        file_name = name_of_file + '.h5'
        return keras.models.load_model(file_name)

    def create_model(self):
        input = Input(shape=(self.max_word_size, self.word_dim))
        using_x = Reshape((self.max_word_size, self.word_dim, 1))(input)

        x1 = Conv2D(self.filter, kernel_size=[2, self.word_dim], activation='relu')(using_x)
        x1 = MaxPooling2D(pool_size=[self.max_word_size - 1, 1], strides=1)(x1)
        x1 = Reshape((1, self.filter))(x1)
        r_result1 = Reshape(target_shape=(self.filter,))(x1)

        x2 = Conv2D(self.filter, kernel_size=[3, self.word_dim], activation='relu')(using_x)
        x2 = MaxPooling2D(pool_size=[self.max_word_size - 2, 1], strides=1)(x2)
        x2 = Reshape((1, self.filter))(x2)
        r_result2 = Reshape(target_shape=(self.filter,))(x2)

        x3 = Conv2D(self.filter, kernel_size=[4, self.word_dim], activation='relu')(using_x)
        x3 = MaxPooling2D(pool_size=[self.max_word_size - 3, 1], strides=1)(x3)
        x3 = Reshape((1, self.filter))(x3)
        r_result3 = Reshape(target_shape=(self.filter,))(x3)

        result = concatenate([r_result1, r_result2, r_result3], axis=1)
        result = Dense(units=self.assign_size)(result)
        result = Activation('softmax', name='predictions')(result)

        model = Model(inputs=input, outputs=result)

        return model

    def compile_model(self, model):
        adam = optimizers.Adam(lr=0.001)
        model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self, model, train_datas, train_labels, test_datas, test_labels, epochs=5, batch_size=5):
        model.fit(train_datas, train_labels, epochs=epochs, batch_size=batch_size
                  , validation_data=(test_datas, test_labels), verbose=False)
        return model
    
    def evaluate_model(self, model, test_datas, test_labels, mode='normal'):
        loss, acc = model.evaluate(test_datas, test_labels)
        if mode == 'normal':
            print('Normal model accurancy: {:5.2f}%'.format(100*acc))
            print('')
        else:
            print(mode, 'mutation operator executed')
            print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
            print('')

    def save_model(self, model, name_of_file, mode='normal'):
        prefix = ''
        file_name = prefix + name_of_file + '.h5'
        model.save(file_name)
        if mode == 'normal':
            print('Normal model is successfully trained and saved at', file_name)
        else: 
            print('Mutated model by ' + mode + ' is successfully saved at', file_name)
        print('')

    def train_and_save_model(self, name_of_file=None, verbose=False):
        (train_datas, train_labels), (test_datas, test_labels) = self.load_data()
        model = self.create_model()
        model = self.compile_model(model)
        model = self.train_model(model, train_datas, train_labels, test_datas. test_labels)

        if verbose:
            print('Current tensorflow version:', tf.__version__)
            print('')

            print('train dataset shape:', train_datas.shape)
            print('test dataset shape:', test_datas.shape)
            print('network architecture:')
            model.summary()
            print('')

            self.evaluate_model(model, test_datas, test_labels)

        self.save_model(model, 'normal_triage_model')