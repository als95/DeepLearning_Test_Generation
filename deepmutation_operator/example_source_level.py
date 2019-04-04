'''
Before usage of this API,
please ensure the following packages are installed.

Tensorflow: 1.11.0
Keras: 2.2.4
NumPy: 1.15.1

Note that you can directly install these packages in ipython notebook
through commands like "!pip install tensorflow==1.11"
'''

# Let's start our demestration
# For this grid, we import some packages and utils.py
import tensorflow as tf
import numpy as np
import keras

import random, math

# You can use the API without creating an utils instance,
# We create an utils instance here for printing some information
# to illustrate that our operators function correctly
import utils
utils = utils.GeneralUtils()

# Prepare training dataset and untrained model for source-level mutation
# Users can their our own dataset and model
import network_triage
network = network_triage.TriageNetwork()

# model is a simple FC(fully-connected) neural network
# dataset is a subset from MNIST dataset with 5000 training data and 1000 testing data
model = network.create_model()
(train_datas, train_labels), (test_datas, test_labels) = network.load_data()

print('train_datas shape:', train_datas.shape)
print('train_labels shape:', train_labels.shape)
print('test_datas shape:', test_datas.shape)
print('test_labels shape:', test_labels.shape)

# Create an instance of source-level mutation operators API
import source_mut_operators
source_mut_opts = source_mut_operators.SourceMutationOperators()

# DR (Data Repetition), see https://github.com/KuoTzu-yang/DeepMutation for more explanation
mutation_ratios = [0.01, 0.1, 0.5]
for mutation_ratio in mutation_ratios:
    (DR_train_datas, DR_train_labels), DR_model = source_mut_opts.DR_mut((train_datas, train_labels), model,
                                                                         mutation_ratio)
    utils.print_messages_SMO('DR', train_datas=train_datas, train_labels=train_labels, mutated_datas=DR_train_datas,
                             mutated_labels=DR_train_labels, mutation_ratio=mutation_ratio)

# LE (Label Error), see https://github.com/KuoTzu-yang/DeepMutation for more explanation
mutation_ratios = [0.01, 0.1, 0.5]
for mutation_ratio in mutation_ratios:
    (LE_train_datas, LE_train_labels), LE_model = source_mut_opts.LE_mut((train_datas, train_labels), model, 0, 16, mutation_ratio)

    mask_equal = LE_train_labels == train_labels
    mask_equal = np.sum(mask_equal, axis=1) == 10
    count_diff = len(train_labels) - np.sum(mask_equal)
    print(len(train_labels))
    print('Mutation ratio:', mutation_ratio)
    print('Number of mislabeled labels:', count_diff)
    print('')

# DM (Data Missing), see https://github.com/KuoTzu-yang/DeepMutation for more explanation
mutation_ratios = [0.01, 0.1, 0.5]
for mutation_ratio in mutation_ratios:
    (DM_train_datas, DM_train_labels), DM_model = source_mut_opts.DM_mut((train_datas, train_labels), model,
                                                                         mutation_ratio)

    utils.print_messages_SMO('DM', train_datas=train_datas, train_labels=train_labels, mutated_datas=DM_train_datas,
                             mutated_labels=DM_train_labels, mutation_ratio=mutation_ratio)

'''
For DF, it's a little difficult to explicitly demonstrate
a large amount of data samples be shuffled.
Here, we simply illustrate how to use DF mutation operator.
'''
# DF (Data Shuffle), see https://github.com/KuoTzu-yang/DeepMutation for more explanation
mutation_ratio = 0.01
(DF_train_datas, DF_train_labels), DF_model = source_mut_opts.DF_mut((train_datas, train_labels), model, mutation_ratio)

# NP (Noise Perturb), see https://github.com/KuoTzu-yang/DeepMutation for more explanation
mutation_ratio = 1
STD = 1
(NP_train_datas, NP_train_labels), NP_model = source_mut_opts.NP_mut((train_datas, train_labels), model, mutation_ratio, STD=STD)

print('A value in the first sample of original dataset', train_datas[0][0])
print('The value after NP mutation', NP_train_datas[0][0])

# Before any mutation on model, let's see the architecture of this model.

# According to the paper, there is a restriction of layer being added or removed.
# The input and output shape of layer being added or removed are required to be same.

# Hence, when you look at the architecture of this model.
# There are layers with same input and output shape in this model for demenstration purpose.
model.summary()

# LR (Layer Removal), see https://github.com/KuoTzu-yang/DeepMutation for more explanation
(LR_train_datas, LR_train_labels), LR_model = source_mut_opts.LR_mut((train_datas, train_labels), model)
LR_model.summary()

# LAs (Layer Addition for source-level mutation), see https://github.com/KuoTzu-yang/DeepMutation for more explanation
(LAs_train_datas, LAs_train_labels), LAs_model = source_mut_opts.LAs_mut((train_datas, train_labels), model)
LAs_model.summary()

'''
For AFRs, it's a little difficult to explicitly demonstrate
Here, we simply illustrate how to use AFRs mutation operator.
'''
# AFRs (Activation Function Removal for source-level mutation), see https://github.com/KuoTzu-yang/DeepMutation for more explanation
(AFRs_train_datas, AFRs_train_labels), AFRs_model = source_mut_opts.AFRs_mut((train_datas, train_labels), model)
AFRs_model.summary()