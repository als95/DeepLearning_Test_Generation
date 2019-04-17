import random as r
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.models import Model


class NeuronCoverage:
    def __init__(self):
        pass

    # ====================================================================================
    #                           gradient ascent process
    # ====================================================================================

    # we compute the gradient of the input bug report wrt this loss
    # Input
    # : x - gradient value of final loss
    # Output
    # : normalized gradient value
    def normalize(self, x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    # ====================================================================================
    #                           neuron coverage prepare process
    # ====================================================================================

    # initiate table to manage activation of neuron
    # Input
    # : model - target model to create table, model_layer_dict - table
    def init_dict(self, model, model_layer_dict):
        for layer in model.layers:
            if 'input' in layer.name or 'concatenate' in layer.name:
                continue
            for index in range(layer.output_shape[-1]):
                model_layer_dict[(layer.name, index)] = False

    # create and initiate 3 table
    # Input
    # : 3 model
    # Output
    # : 3 table
    def init_coverage_tables(self, model1, model2, model3):
        model_layer_dict1 = defaultdict(bool)
        model_layer_dict2 = defaultdict(bool)
        model_layer_dict3 = defaultdict(bool)
        self.init_dict(model1, model_layer_dict1)
        self.init_dict(model2, model_layer_dict2)
        self.init_dict(model3, model_layer_dict3)
        return model_layer_dict1, model_layer_dict2, model_layer_dict3

    # choice neuron don't activate
    # Input
    # : model_layer_dict - table to manage activation of neuron
    # Output
    # : layer_name - layer name(index) of choose neuron, index - index of choose neuron in layer
    def neuron_to_cover(self, model_layer_dict):
        not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
        if not_covered:
            layer_name, index = r.choice(not_covered)
        else:
            layer_name, index = r.choice(list(model_layer_dict.keys()))
        return layer_name, index

    # calculate neuron coverage
    # Input
    # : model_layer_dict - table to manage activation of neuron
    # Output
    # : covered_neurons - # of activated neurons, total_neurons - # of total neurons
    #   , neuron coverage
    def neuron_covered(self, model_layer_dict):
        covered_neurons = len([v for v in model_layer_dict.values() if v])
        total_neurons = len(model_layer_dict)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    # create and initiate table to store maximum and minimum neuron output
    # Input
    # : 3 model, input_datas - test data set of model
    # Output
    # : 3 maximum table, 3 minimum table
    def init_neuron_threshold_tables(self, model1, model2, model3, input_datas):
        model_max_threshold1 = defaultdict(float)
        model_max_threshold2 = defaultdict(float)
        model_max_threshold3 = defaultdict(float)
        model_min_threshold1 = defaultdict(float)
        model_min_threshold2 = defaultdict(float)
        model_min_threshold3 = defaultdict(float)

        self.init_neuron_threshold(input_datas, model1, model_max_threshold1, model_min_threshold1)
        self.init_neuron_threshold(input_datas, model2, model_max_threshold2, model_min_threshold2)
        self.init_neuron_threshold(input_datas, model3, model_max_threshold3, model_min_threshold3)
        return model_max_threshold1, model_max_threshold2, model_max_threshold3 \
            , model_min_threshold1, model_min_threshold2, model_min_threshold3

    # initiate table to store maximum and minimum neuron output
    # Input
    # : input_datas - test data set of model, model_max_threshold - table to store maximum neuron output
    #   , model_min_threshold - table to store minimum neuron output
    def init_neuron_threshold(self, input_datas, model, model_max_threshold, model_min_threshold):
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

    # ====================================================================================
    #                           neuron coverage calculate process
    # ====================================================================================

    # scale neuron output in same layer
    # Input
    # : intermediate_layer_output - target layer to scale
    # Output
    # : X_scaled - scaled layer output
    def scale(self, intermediate_layer_output, rmax=1, rmin=0):
        X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
                intermediate_layer_output.max() - intermediate_layer_output.min())
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled

    # update neuron coverage
    # Input
    # : input_data - test data set of model, model_layer_dict - table to manage activation of neuron
    #   , max,min_threshold_dict - table to store maximum,minimum neuron output, coverage_name - option of coverage
    def update_coverage(self, input_data, model, model_layer_dict, max_threshold_dict, min_threshold_dict, coverage_name,
                        threshold):
        layer_names = [layer.name for layer in model.layers
                       if 'concatenate' not in layer.name and 'input' not in layer.name]

        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
        intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

        if coverage_name == 'dxp':
            for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
                scaled = self.scale(intermediate_layer_output[0])
                for num_neuron in range(scaled.shape[-1]):
                    if np.mean(scaled[..., num_neuron]) > threshold \
                            and not model_layer_dict[(layer_names[i], num_neuron)]:
                        model_layer_dict[(layer_names[i], num_neuron)] = True
        elif coverage_name == 'kmnc':
            for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
                layer_output = intermediate_layer_output[0]
                for num_neuron in range(layer_output.shape[-1]):
                    if np.mean(layer_output[..., num_neuron]) < max_threshold_dict[(layer_names[i], num_neuron)] \
                            and np.mean(
                        layer_output[..., num_neuron] > min_threshold_dict[(layer_names[i], num_neuron)]) \
                            and not model_layer_dict[(layer_names[i], num_neuron)]:
                        model_layer_dict[(layer_names[i], num_neuron)] = True
        elif coverage_name == 'snbc':
            for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
                layer_output = intermediate_layer_output[0]
                for num_neuron in range(layer_output.shape[-1]):
                    if np.mean(layer_output[..., num_neuron]) >= max_threshold_dict[(layer_names[i], num_neuron)] \
                            and not model_layer_dict[(layer_names[i], num_neuron)]:
                        model_layer_dict[(layer_names[i], num_neuron)] = True