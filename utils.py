import random as r
import codecs
from collections import defaultdict

from sklearn import preprocessing
import numpy as np
from keras import backend as K
from keras.models import Model
from random import *
from vocabulary.vocabulary import Vocabulary as vb
from nltk.corpus import wordnet

from Model import WordPreprocessor, WordVecModeler

import json


def data_preprocess(raw_x):
    word_dim = 50
    sum_vec = 1
    word_vec_modeler = WordVecModeler(dim=word_dim)
    max_word = 500

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


def data_train(vec_x, resize_vec, one_hot_y):
    test_percent = 0.1

    train_vec_x = resize_vec[:int(len(vec_x) * (1 - test_percent))]
    train_one_hot_y = one_hot_y[:int(len(one_hot_y) * (1 - test_percent))]

    return train_vec_x, train_one_hot_y


def data_test(vec_x, resize_vec, one_hot_y):
    test_percent = 0.1

    test_vec_x = resize_vec[int(len(vec_x) * (1 - test_percent)):]
    test_one_hot_y = one_hot_y[int(len(one_hot_y) * (1 - test_percent)):]

    return test_vec_x, test_one_hot_y


def data_deprocess(choice_words):
    str = []
    for i in range(len(choice_words)):
        str.append(" " + choice_words[i])
    return ''.join(str)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constarint_synonym(gen, grads_value, args):
    if args.test_generation == 'fgsm':
        new_grads = np.ones_like(grads_value)
        grad_mean = np.mean(grads_value)
        grads = grad_mean * new_grads

        iterate = np.amax(grads)
        iterate = iterate * 1000

        syns = []
        i = 0
        j = 0
        return_type = False

        while j < iterate:
            while len(syns) == 0 and i < len(gen):
                i += 1
                index = randint(0, len(gen) - 1)
                syns = wordnet.synsets(str(gen[index]))

            if len(gen) != i:
                gen[index] = syns[0].name()
                return_type = True
            j += 1

        return gen, return_type


def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'input' in layer.name or 'concatenate' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = r.choice(not_covered)
    else:
        layer_name, index = r.choice(list(model_layer_dict.keys()))
    return layer_name, index


def neuron_covered(model_layer_dict, args):
    if args.converage == 'dxp':
        covered_neurons = len([v for v in model_layer_dict.values() if v])
        total_neurons = len(model_layer_dict)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def init_neuron_threshold(model, model_threshold):
    for layer in model.layers:
        if 'input' in layer.name or 'concatenate' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_threshold[(layer.name, index)] = 0


def init_neuron_threshold_tables(model1, model2, model3):
    model_threshold1 = defaultdict(bool)
    model_threshold2 = defaultdict(bool)
    model_threshold3 = defaultdict(bool)
    init_neuron_threshold(model1, model_threshold1)
    init_neuron_threshold(model2, model_threshold2)
    init_neuron_threshold(model3, model_threshold3)
    return model_threshold1, model_threshold2, model_threshold3


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def update_coverage(input_data, model, model_layer_dict, args):
    if args.converage == 'dxp':
        layer_names = [layer.name for layer in model.layers if
                       'concatenate' not in layer.name and 'input' not in layer.name]

        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
        intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            scaled = scale(intermediate_layer_output[0])
            for num_neuron in range(scaled.shape[-1]):
                if np.mean(scaled[..., num_neuron]) > args.threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                    model_layer_dict[(layer_names[i], num_neuron)] = True


def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False
