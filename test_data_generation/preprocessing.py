import codecs
from .w2v import WordPreprocessor, WordVecModel
import numpy as np
from sklearn import preprocessing


class Preprocessing:
    def __init__(self, word_dim):
        word_preprocessor = WordPreprocessor()
        self.raw_x = word_preprocessor.data_preprocessing('../input/test_data_generation/input/desc.txt')
        assign_file = codecs.open('../input/test_data_generation/input/assignTo.txt', "r", "utf-8")

        self.word_vec_modeler = WordVecModel(dim=word_dim)
        # self.word_vec_modeler.load_word_vec("word_vec_dim_50_skip_window5_nostacktrace")
        self.word_vec_modeler.load_word_vec("../input/GoogleNews-vectors-negative300.bin")

        self.raw_y = []
        for line in assign_file:
            line = line.strip()
            self.raw_y.append(line)
        # load data end
        le = preprocessing.LabelEncoder()
        enc = preprocessing.OneHotEncoder()
        le.fit(self.raw_y)
        self.assign_num = len(set(self.raw_y))
        y_to_number = np.array(le.transform(self.raw_y))
        y_to_number = np.reshape(y_to_number, [-1, 1])
        enc.fit(y_to_number)
        self.one_hot_y = enc.transform(y_to_number).toarray()
        # assign one_hot encoding

        self.vec_x, self.resize_vec = self.data_preprocess(self.raw_x)
        pass

    def data_preprocess(self, raw_x):
        sum_vec = 1
        word_dim = 300
        max_word = 500
        vec_x = None
        vec_size = []
        for idx, doc in enumerate(raw_x):
            temp_arr = []
            for word_idx, word in enumerate(doc):
                if self.word_vec_modeler.get_vector_from_word(word) is not None:
                    temp_arr.append(self.word_vec_modeler.get_vector_from_word(word))
                if word_idx == max_word - 1:
                    break

            if len(temp_arr) == 0:
                temp_arr.append(np.zeros(300))

            if len(temp_arr) < max_word:
                for _ in range(max_word - len(temp_arr)):
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

        return vec_x, resize_vec

    def data_deprocess(self, words):
        result = []
        for i in range(len(words)):
            result.append(" " + words[i])
        return ''.join(result)

    def generate_train_data(self, test_percent):
        x_length = int(len(self.vec_x) * (1 - test_percent))
        y_length = int(len(self.one_hot_y) * (1 - test_percent))

        train_raw_x = self.raw_x[:x_length]
        train_raw_y = self.raw_y[:y_length]
        train_vec_x = self.resize_vec[:x_length]
        train_one_hot_y = self.one_hot_y[:y_length]

        return train_raw_x, train_raw_y, train_vec_x, train_one_hot_y

    def generate_test_data(self, test_percent):
        x_length = int(len(self.vec_x) * (1 - test_percent))
        y_length = int(len(self.one_hot_y) * (1 - test_percent))

        test_raw_x = self.raw_x[x_length:]
        test_raw_y = self.raw_y[y_length:]
        test_vec_x = self.resize_vec[x_length:]
        test_one_hot_y = self.one_hot_y[y_length:]

        return test_raw_x, test_raw_y, test_vec_x, test_one_hot_y
