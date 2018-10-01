import tensorflow as tf
import numpy as np
from Model import WordPreprocessor, WordVecModeler
from sklearn import preprocessing
import codecs

class Trainer:
    def __init__(self, word_dim, assign_size,TB_SUMMARY_DIR,filter=128, max_word_size=500, stack=2):
        self.word_dim = word_dim
        self.assign_size = assign_size
        self.TB_SUMMARY_DIR = TB_SUMMARY_DIR
        self.max_word_size = int(max_word_size)
        self.stack = stack
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.max_word_size, self.word_dim])
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, assign_size])
        self.keep_prob = 1.0
        using_x = tf.reshape(self.X,[-1,self.max_word_size,word_dim,1])
        conv1= tf.layers.conv2d(using_x,filters=filter,kernel_size=[2,word_dim],activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[max_word_size-1,1],strides=1)
        result1 = tf.reshape(pool1, (-1, 1, filter))
        print("result1 : ", result1.shape)
        self.r_result1 = tf.reshape(result1, (-1, filter))
        print("r_result1 : ", self.r_result1.shape)

        conv2 = tf.layers.conv2d(using_x, filters=filter, kernel_size=[3, word_dim],activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[max_word_size-2, 1], strides=1)
        result2 = tf.reshape(pool2, (-1, 1, filter))
        self.r_result2 = tf.reshape(result2, (-1, filter))

        conv3 = tf.layers.conv2d(using_x, filters=filter, kernel_size=[4, word_dim],activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[max_word_size-3, 1], strides=1)
        result3 = tf.reshape(pool3, (-1, 1, filter))
        self.r_result3 = tf.reshape(result3, (-1, filter))

        self.treat = tf.concat([self.r_result1,self.r_result2,self.r_result3],axis=1)
        print("treat : ", self.treat.shape)
        self.logits = tf.layers.dense(inputs=self.treat,units=assign_size)
        print("logits : ", self.logits.shape)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        cost_summ = tf.summary.scalar("cost", cost)
        train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        predict = tf.argmax(self.logits, 1)
        print("predict : ", predict.shape)
        correct_predict = tf.equal(predict, tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        accuracy_summ = tf.summary.scalar("accuracy", accuracy)
        self.cost_summ = cost_summ
        self.accuracy_summ = accuracy_summ
        self.cost = cost
        self.train = train
        self.predict = predict
        self.accuracy = accuracy
        self.saver = tf.train.Saver()

    def train_op(self, train_x, train_y, test_x,
                 test_y, epoch=5, batch_size=5):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.45
        with tf.Session(config=config) as sess:
            cost_writer = tf.summary.FileWriter(self.TB_SUMMARY_DIR + "/cost")
            accuracy_writer = tf.summary.FileWriter(self.TB_SUMMARY_DIR + "/accuracy")
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                random_idx = np.random.permutation(len(train_x) // batch_size)
                for j in range(len(random_idx)):
                    batch_x = train_x[random_idx[j] * batch_size:(random_idx[j] + 1) * batch_size]
                    batch_y = train_y[random_idx[j] * batch_size:(random_idx[j] + 1) * batch_size]
                    self.keep_prob = 0.7
                    _ = sess.run(self.train, feed_dict={self.X: batch_x, self.Y: batch_y})
                    if j % 10 == 0:
                        summary, cost = sess.run([self.cost_summ, self.cost],
                                                 feed_dict={self.X: batch_x, self.Y: batch_y})
                        cost_writer.add_summary(summary, i * len(random_idx) + j)
                        print('epoch {0}: step={1}, cost = {2}'.format(i, j, cost))
                self.keep_prob = 1.0
                accuracy_summ, acc = sess.run([self.accuracy_summ, self.accuracy],
                                              feed_dict={self.X: test_x, self.Y: test_y})
                accuracy_writer.add_summary(accuracy_summ, i * len(random_idx))
                print('epoch {0}: accuracy={1}'.format(i, acc))
            self.saver.save(sess, './save_model.ckpt')

    def test(self, test_x, test_y):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.45
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, './save_model.ckpt')
            self.keep_prob = 1.0
            accuracy = np.mean(np.argmax(test_y, axis=1) == sess.run(self.predict,
                                                                     feed_dict={self.X: test_x, self.Y: test_y}))
        return accuracy

    def myop(self,test_x,test_y):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            r1 = sess.run(self.r_result1,feed_dict={self.X: test_x, self.Y: test_y})
            print(np.shape(r1))

if __name__ == '__main__':
    word_dim = 50
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
    # 데이터 불러오기 끝
    le = preprocessing.LabelEncoder()
    enc = preprocessing.OneHotEncoder()
    le.fit(raw_y)
    assign_num = len(set(raw_y))
    y_to_number = np.array(le.transform(raw_y))
    y_to_number = np.reshape(y_to_number, [-1, 1])
    enc.fit(y_to_number)
    one_hot_y = enc.transform(y_to_number).toarray()
    # assign one_hot 인코딩

    word_vec_modeler.load_word_vec("word_vec_dim_50_skip_window5_nostacktrace")
    vec_x = None
    vec_size = []
    for idx, doc in enumerate(raw_x):
        temp_arr = []
        for word_idx, word in enumerate(doc):
            if word_vec_modeler.get_vector_from_word(word) is not None:
                temp_arr.append(word_vec_modeler.get_vector_from_word(word))
            if word_idx == max_word - 1:
                break
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
    print("train : ", train_one_hot_y.shape)

    hs_trainer = Trainer(assign_size=assign_num, word_dim=word_dim, max_word_size=500 / sum_vec,
                         TB_SUMMARY_DIR=TB_SUMMARY_DIR, filter=512)
    # hs_trainer.my_op(train_vec_x,train_one_hot_y)
    hs_trainer.train_op(train_x=train_vec_x, train_y=train_one_hot_y, test_x=test_vec_x,
                        test_y=test_one_hot_y, batch_size=4, epoch=50)
    print('accuracy', hs_trainer.test(test_vec_x, test_one_hot_y))