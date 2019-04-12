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

# Prepare training dataset and untrained model
# Users can their our own dataset and model
import network_triage
network = network_triage.TriageNetwork()

# model is a simple FC(fully-connected) neural network
# dataset is a subset from MNIST dataset with 5000 training data and 1000 testing data
(train_datas, train_labels), (test_datas, test_labels) = network.load_data()
model = network.create_model()

print('train_datas shape:', train_datas.shape)
print('train_labels shape:', train_labels.shape)
print('test_datas shape:', test_datas.shape)
print('test_labels shape:', test_labels.shape)

# Compile and train our example model
model = network.compile_model(model)
model = network.train_model(model, train_datas, train_labels, test_datas, test_labels)

# Let's print the accurancy of our example model to see whether it's trained
# You should see accurancy around 92%
network.evaluate_model(model, test_datas, test_labels)

# Create an instance of source-level mutation operators API
import model_mut_operators
model_mut_opts = model_mut_operators.ModelMutationOperators()

# GF (Gaussian Fuzzing), see https://github.com/KuoTzu-yang/DeepMutation for more explanation
# Notice you should re-compile the mutated model
mutation_ratio = 0.1
STD=0.2
GF_model = model_mut_opts.GF_mut(model, mutation_ratio, STD=STD)
GF_model = network.compile_model(GF_model)

'''
Let's evaluate the performance of the mutated model compared to the original model. 
Either increase of mutation_ratio or STD will give rise to a decrease in accuracy, which is consistent with our expectation.  
You can turn the value of mutation_ratio and STD to observe the difference of accuracy via the print function. 
'''
utils.print_messages_MMM_generators('GF', network=network, test_datas=test_datas, test_labels=test_labels, model=model,
                                    mutated_model=GF_model, STD=STD, mutation_ratio=mutation_ratio)

'''
WS (Weight Shuffling), see https://github.com/KuoTzu-yang/DeepMutation for more explanation
Users should re-compile the mutated model. 
For WS, we simply demostrate the usage of WS mutation operator. 
'''
mutation_ratio = 0.1
WS_model = model_mut_opts.WS_mut(model, mutation_ratio)
WS_model = network.compile_model(WS_model)

'''
NEB (Neuron Effect Block), see https://github.com/KuoTzu-yang/DeepMutation for more explanation
Users should re-compile the mutated model. 
For NEB, we simply demostrate the usage of NEB mutation operator. 
'''
mutation_ratio = 0.1
NEB_model = model_mut_opts.NEB_mut(model, mutation_ratio)
NEB_model = network.compile_model(NEB_model)

'''
NAI (Neuron Activation Inverse), see https://github.com/KuoTzu-yang/DeepMutation for more explanation
Users should re-compile the mutated model. 
For NAI, we simply demostrate the usage of NAI mutation operator. 
'''
mutation_ratio = 0.1
NAI_model = model_mut_opts.NAI_mut(model, mutation_ratio)
NAI_model = network.compile_model(NAI_model)

'''
NS (Neuron Switch), see https://github.com/KuoTzu-yang/DeepMutation for more explanation
Users should re-compile the mutated model. 
For NS, we simply demostrate the usage of NS mutation operator. 
'''
mutation_ratio = 0.1
NS_model = model_mut_opts.NS_mut(model, mutation_ratio)
NS_model = network.compile_model(NS_model)

'''
LD (Layer Deactivation), see https://github.com/KuoTzu-yang/DeepMutation for more explanation
Users should re-compile the mutated model. 
'''
LD_model = model_mut_opts.LD_mut(model)
LD_model = network.compile_model(LD_model)
utils.print_messages_MMM_generators('LD', network=network, test_datas=test_datas, test_labels=test_labels, model=model,
                                    mutated_model=LD_model)

'''
LAm (Layer Addition for model-level), see https://github.com/KuoTzu-yang/DeepMutation for more explanation
Users should re-compile the mutated model. 
'''
LAm_model = model_mut_opts.LAm_mut(model)
LAm_model = network.compile_model(LAm_model)
utils.print_messages_MMM_generators('LAm', network=network, test_datas=test_datas, test_labels=test_labels, model=model, mutated_model=LAm_model)

'''
AFRm (Activation Function Removal for model-level), see https://github.com/KuoTzu-yang/DeepMutation for more explanation
Users should re-compile the mutated model. 
'''
AFRm_model = model_mut_opts.AFRm_mut(model)
AFRm_model = network.compile_model(AFRm_model)
utils.print_messages_MMM_generators('AFRm', network=network, test_datas=test_datas, test_labels=test_labels, model=model, mutated_model=AFRm_model)


