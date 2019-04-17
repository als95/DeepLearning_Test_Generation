import tensorflow as tf
from .configs import bcolors
from keras import optimizers
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, concatenate, Dense, Activation, Input, Reshape
from .preprocessing import Preprocessing
import argparse


class TriageModel:
    def __init__(self, word_dim, assign_size, filter=128, max_word_size=500, stack=2, model_id=1):
        self.model_id = model_id
        self.word_dim = int(word_dim)
        self.assign_size = assign_size
        self.max_word_size = int(max_word_size)
        self.stack = stack
        self.input = Input(shape=(self.max_word_size, self.word_dim), name='input')
        using_x = Reshape((self.max_word_size, self.word_dim, 1), name='reshape_using')(self.input)

        x1 = Conv2D(filter, kernel_size=[2, word_dim], activation='relu', name='block1_conv1')(using_x)
        x1 = MaxPooling2D(pool_size=[max_word_size - 1, 1], strides=1, name='block1_pool1')(x1)
        x1 = Reshape((1, filter), name='reshape_x1')(x1)
        self.r_result1 = Reshape(target_shape=(filter,), name='reshape_x1_filter')(x1)

        x2 = Conv2D(filter, kernel_size=[3, word_dim], activation='relu', name='block2_conv2')(using_x)
        x2 = MaxPooling2D(pool_size=[max_word_size - 2, 1], strides=1, name='block2_pool2')(x2)
        x2 = Reshape((1, filter), name='reshape_x2')(x2)
        self.r_result2 = Reshape(target_shape=(filter,), name='reshape_x2_filter')(x2)

        x3 = Conv2D(filter, kernel_size=[4, word_dim], activation='relu', name='block3_conv3')(using_x)
        x3 = MaxPooling2D(pool_size=[max_word_size - 3, 1], strides=1, name='block3_pool3')(x3)
        x3 = Reshape((1, filter), name='reshape_x3')(x3)
        self.r_result3 = Reshape(target_shape=(filter,), name='reshape_x3_filter')(x3)

        self.result = concatenate([self.r_result1, self.r_result2, self.r_result3], axis=1, name='concatenate')
        self.result = Dense(units=assign_size, name='before_softmax')(self.result)
        self.result = Activation('softmax', name='predictions')(self.result)

        self.model = Model(inputs=self.input, outputs=self.result)

    def train_nonlearning(self, train_x, train_y, test_x, test_y, epoch=5, batch_size=5):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.45

        adam = optimizers.Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), batch_size=batch_size, epochs=epoch)
        self.model.save_weights('./Model' + self.model_id + '.h5')
        score = self.model.evaluate(test_x, test_y, verbose=0)
        print('\n')
        print('Overall Test score: ', score[0])
        print('Overall Test accuracy: ', score[1])

    def train_learning(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.45

        self.model.load_weights('./Model' + self.model_id + '.h5')
        print(bcolors.OKBLUE + 'Model' + self.model_id + ' loaded' + bcolors.ENDC)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Triage Model training')
    parser.add_argument('model_id', help="model id", default=1, type=int)
    args = parser.parse_args()

    word_dim = 300
    test_percent = 0.1
    sum_vec = 1

    preprocessing = Preprocessing(word_dim)
    assign_num = preprocessing.assign_num

    train_raw_x, train_raw_y, train_vec_x, train_one_hot_y = preprocessing.generate_train_data(test_percent)
    test_raw_x, test_raw_y, test_vec_x, test_one_hot_y = preprocessing.generate_test_data(test_percent)

    triage_model = TriageModel(assign_size=assign_num, word_dim=word_dim, max_word_size=int(500/sum_vec), filter=512,
                               model_id=args.model_id)

    triage_model.train_nonlearning(train_x=train_vec_x, train_y=train_one_hot_y, test_x=test_vec_x,
                                   test_y=test_one_hot_y, epoch=50, batch_size=4)
