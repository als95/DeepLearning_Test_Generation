import random as r
import codecs
from collections import defaultdict

from sklearn import preprocessing
import numpy as np
from keras import backend as K
from keras.models import Model
from random import *

from nltk.corpus import wordnet

from Model import WordPreprocessor, WordVecModeler

import json


def data_preprocess(raw_x, word_vec_modeler):
    sum_vec = 1
    word_dim = 300
    max_word = 500
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


def constraint_add(gen, grads_value, args, raw_x):
    max_word = 1000
    x = randint(0, len(raw_x) - 1)
    y = randint(0, len(raw_x[x]) - 1)
    if args.test_generation == 'basic':
        iterate = np.amax(grads_value)

        return_type = False

        j = 0
        iterate += 1
        while j < iterate and len(gen) < max_word:
            gen.append(raw_x[x][y])
            x = randint(0, len(raw_x) - 1)
            y = randint(0, len(raw_x[x]) - 1)
            j += 1
            return_type = True

        pass
    elif args.test_generation == 'dxp':
        iterate = np.mean(grads_value)

        return_type = False

        if iterate >= 1 and len(gen) < max_word:
            gen.append(raw_x[x][y])
            return_type = True

        pass
    elif args.test_generation == 'fgsm':
        iterate = np.mean(grads_value)
        iterate = np.sign(iterate)

        return_type = False

        if iterate > 0 and len(gen) < max_word:
            gen.append(raw_x[x][y])
            return_type = True

        pass

    return gen, return_type


def constarint_synonym(gen, grads_value, args):
    if args.test_generation == 'basic':
        iterate = np.mean(grads_value)

        syns_list = []
        return_type = False

        j = 0
        i = r.randint(0, len(gen) - 1)
        while j < iterate and i < len(gen) and return_type is False:
            while len(syns_list) == 0 and i < len(gen) and return_type is False:
                syns_list = wordnet.synsets(gen[i])
                if len(syns_list) != 0:
                    break
                # if vb.synonym(gen[i]):
                #     syns_list = json.loads(vb.synonym(gen[i]))
                #     break
                i += 1

            if len(gen) > i and len(syns_list) != 0:
                # syns = syns_list[0]
                # gen[i - 1] = syns['text']
                syns_index = 0
                while True:
                    if len(syns_list) <= syns_index:
                        return_type = False
                        break
                    syns = syns_list[syns_index].lemmas()[0].name()
                    syns_index += 1
                    if syns != gen[i]:
                        return_type = True
                        gen[i] = syns
                        break
            syns_list = []
            j += 1
            i = r.randint(0, len(gen) - 1)

        return gen, return_type

    elif args.test_generation == 'dxp':
        iterate = np.mean(grads_value)

        syns_list = []
        k = 0
        j = 0
        i = r.randint(0, len(gen) - 1)
        return_type = False

        while j < iterate and k < len(gen) and return_type is False:
            while len(syns_list) == 0 and i < len(gen) and return_type is False:
                syns_list = wordnet.synsets(gen[i])
                if len(syns_list) != 0:
                    break

                # if vb.synonym(gen[i]):
                #     syns_list = json.loads(vb.synonym(gen[i]))
                #     break
                i += 1

            if len(gen) > i and len(syns_list) != 0:
                syns_index = 0
                while True:
                    if len(syns_list) <= syns_index:
                        return_type = False
                        break
                    syns = syns_list[syns_index].lemmas()[0].name()
                    syns_index += 1
                    if syns != gen[i]:
                        return_type = True
                        gen[i] = syns
                        break
            syns_list = []
            i = r.randint(0, len(gen) - 1)
            j += 1
            k += 1


        return gen, return_type
    elif args.test_generation == 'fgsm':
        grad_mean = np.mean(grads_value)

        iterate = np.sign(grad_mean)

        syns_list = []
        k = 0
        j = 0
        i = r.randint(0, len(gen) - 1)
        return_type = False

        while j < iterate and k < len(gen) and return_type is False:
            while len(syns_list) == 0 and i < len(gen) and return_type is False:
                syns_list = wordnet.synsets(gen[i])
                if len(syns_list) != 0:
                    break

                # if vb.synonym(gen[i]):
                #     syns_list = json.loads(vb.synonym(gen[i]))
                #     break
                i += 1

            if len(gen) > i and len(syns_list) != 0:
                syns_index = 0
                while True:
                    if len(syns_list) <= syns_index:
                        return_type = False
                        break
                    syns = syns_list[syns_index].lemmas()[0].name()
                    syns_index += 1
                    if syns != gen[i]:
                        return_type = True
                        gen[i] = syns
                        break
            syns_list = []
            i = r.randint(0, len(gen) - 1)
            j += 1
            k += 1

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


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def init_neuron_threshold(input_datas, model, model_max_threshold, model_min_threshold):
    for layer in model.layers:
        if 'input' in layer.name or 'concatenate' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_max_threshold[(layer.name, index)] = 0
            model_min_threshold[(layer.name, index)] = 0

    layer_names = [layer.name for layer in model.layers
                   if 'concatenate' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                    outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    for input_data in input_datas:
        intermediate_layer_output = intermediate_layer_model.predict(np.expand_dims(input_data, axis=0))

        for j, intermediate_neuron_outputs in enumerate(intermediate_layer_output):
            layer_output = intermediate_neuron_outputs[0]
            for num_neuron in range(layer_output.shape[-1]):
                if np.mean(layer_output[..., num_neuron]) > model_max_threshold[(layer_names[j], num_neuron)]:
                    model_max_threshold[(layer_names[j], num_neuron)] = np.mean(layer_output[..., num_neuron])
                if np.mean(layer_output[..., num_neuron]) < model_min_threshold[(layer_names[j], num_neuron)]:
                    model_min_threshold[(layer_names[j], num_neuron)] = np.mean(layer_output[..., num_neuron])


def init_neuron_threshold_tables(model1, model2, model3, input_datas):
    model_max_threshold1 = defaultdict(float)
    model_max_threshold2 = defaultdict(float)
    model_max_threshold3 = defaultdict(float)
    model_min_threshold1 = defaultdict(float)
    model_min_threshold2 = defaultdict(float)
    model_min_threshold3 = defaultdict(float)

    init_neuron_threshold(input_datas, model1, model_max_threshold1, model_min_threshold1)
    init_neuron_threshold(input_datas, model2, model_max_threshold2, model_min_threshold2)
    init_neuron_threshold(input_datas, model3, model_max_threshold3, model_min_threshold3)
    return model_max_threshold1, model_max_threshold2, model_max_threshold3\
        , model_min_threshold1, model_min_threshold2, model_min_threshold3


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def update_coverage(input_data, model, model_layer_dict, max_threshold_dict, min_threshold_dict, args):
    layer_names = [layer.name for layer in model.layers
                   if 'concatenate' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    if args.coverage == 'dxp':
        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            scaled = scale(intermediate_layer_output[0])
            for num_neuron in range(scaled.shape[-1]):
                if np.mean(scaled[..., num_neuron]) > args.threshold \
                        and not model_layer_dict[(layer_names[i], num_neuron)]:
                    model_layer_dict[(layer_names[i], num_neuron)] = True
    elif args.coverage == 'kmnc':
        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            layer_output = intermediate_layer_output[0]
            for num_neuron in range(layer_output.shape[-1]):
                if np.mean(layer_output[..., num_neuron]) < max_threshold_dict[(layer_names[i], num_neuron)]\
                    and np.mean(layer_output[..., num_neuron] > min_threshold_dict[(layer_names[i], num_neuron)])\
                        and not model_layer_dict[(layer_names[i], num_neuron)]:
                    model_layer_dict[(layer_names[i], num_neuron)] = True
    elif args.coverage == 'snbc':
        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            layer_output = intermediate_layer_output[0]
            for num_neuron in range(layer_output.shape[-1]):
                if np.mean(layer_output[..., num_neuron]) >= max_threshold_dict[(layer_names[i], num_neuron)]\
                    and not model_layer_dict[(layer_names[i], num_neuron)]:
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
