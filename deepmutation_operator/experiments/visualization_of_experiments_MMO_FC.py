# Basic setup & Import related modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

import random, math

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import utils, network, model_mut_operators
utils = utils.GeneralUtils()
network = network.FCNetwork()
model_mut_opts = model_mut_operators.ModelMutationOperators()

(train_datas, train_labels), (test_datas, test_labels) = network.load_data()

print('train_datas shape:', train_datas.shape)
print('test_datas shape:', test_datas.shape)
print('train_labels shape:', train_labels.shape)
print('test_labels shape:', test_labels.shape)

mutation_ratios = [i*0.05 + 0.05 for i in range(20)]

model = network.create_normal_FC_model()
model = network.compile_model(model)
trained_model = network.train_model(model, train_datas, train_labels)

loss, acc = trained_model.evaluate(test_datas, test_labels, verbose=True)
print('loss:', loss)
print('accurancy:', acc)


# GF (Gaussian Fuzzing)
def GF_experiment(STD, lower_bound, upper_bound):
    normal_accs = []
    mutant_accs_normal = []
    mutant_accs_uniform = []

    for mutation_ratio in mutation_ratios:
        trained_mutated_model_normal = model_mut_opts.GF_mut(trained_model, mutation_ratio, prob_distribution='normal',
                                                             STD=STD)
        trained_mutated_model_normal = network.compile_model(trained_mutated_model_normal)
        trained_mutated_model_uniform = model_mut_opts.GF_mut(trained_model, mutation_ratio,
                                                              prob_distribution='uniform', lower_bound=lower_bound,
                                                              upper_bound=upper_bound)
        trained_mutated_model_uniform = network.compile_model(trained_mutated_model_uniform)

        loss, acc = trained_model.evaluate(test_datas, test_labels, verbose=False)
        normal_accs.append(acc)
        mutant_loss, mutant_acc = trained_mutated_model_normal.evaluate(test_datas, test_labels, verbose=False)
        mutant_accs_normal.append(mutant_acc)
        mutant_loss, mutant_acc = trained_mutated_model_uniform.evaluate(test_datas, test_labels, verbose=False)
        mutant_accs_uniform.append(mutant_acc)

    return normal_accs, mutant_accs_normal, mutant_accs_uniform


normal_accs_1, mutant_accs_normal_1, mutant_accs_uniform_1 = GF_experiment(0.1, -0.1, 0.1)
normal_accs_2, mutant_accs_normal_2, mutant_accs_uniform_2 = GF_experiment(0.5, -0.5, 0.5)
normal_accs_3, mutant_accs_normal_3, mutant_accs_uniform_3 = GF_experiment(1, -1, 1)

set_of_normal_accs = [normal_accs_1, normal_accs_2, normal_accs_3]
set_of_mutant_accs_normal = [mutant_accs_normal_1, mutant_accs_normal_2, mutant_accs_normal_3]
set_of_mutant_accs_uniform = [mutant_accs_uniform_1, mutant_accs_uniform_2, mutant_accs_uniform_3]
STDs = ['0.1', '0.5', '1']

plt.figure(figsize=(10, 20))

for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.axis([0, 1, 0, 1])
    plt.plot(mutation_ratios, set_of_normal_accs[i])
    plt.plot(mutation_ratios, set_of_mutant_accs_normal[i])
    plt.plot(mutation_ratios, set_of_mutant_accs_uniform[i])
    plt.legend(['normal accurancy', 'mutant accurancy with normal distribution', 'mutant accurancy with uniform distribution'], loc='center right')
    plt.title('GF mutation accurancy with STD as ' + STDs[i] + ' and range [-' + STDs[i] + ', ' + STDs[i] + ']')
    plt.xlabel('Mutation ratio')
    plt.ylabel('Accurancy')

plt.show()


def WS_experiment():
    normal_accs = []
    mutant_accs = []

    for mutation_ratio in mutation_ratios:
        trained_mutated_model = model_mut_opts.WS_mut(trained_model, mutation_ratio)
        trained_mutated_model = network.compile_model(trained_mutated_model)

        loss, acc = trained_model.evaluate(test_datas, test_labels, verbose=False)
        normal_accs.append(acc)
        mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
        mutant_accs.append(mutant_acc)

    return normal_accs, mutant_accs


normal_accs, mutant_accs = WS_experiment()

plt.axis([0, 1, 0, 1])
plt.plot(mutation_ratios, normal_accs)
plt.plot(mutation_ratios, mutant_accs)

plt.legend(['normal accurancy', 'mutant accurancy'], loc='lower left')
plt.title('WS mutation accurancy')
plt.xlabel('Mutation ratio')
plt.ylabel('Accurancy')
plt.show()


def NEB_experiment():
    normal_accs = []
    mutant_accs = []

    for mutation_ratio in mutation_ratios:
        trained_mutated_model = model_mut_opts.NEB_mut(trained_model, mutation_ratio)
        trained_mutated_model = network.compile_model(trained_mutated_model)

        loss, acc = trained_model.evaluate(test_datas, test_labels, verbose=False)
        normal_accs.append(acc)
        mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
        mutant_accs.append(mutant_acc)

    return normal_accs, mutant_accs


normal_accs, mutant_accs = NEB_experiment()

plt.axis([0, 1, 0, 1])
plt.plot(mutation_ratios, normal_accs)
plt.plot(mutation_ratios, mutant_accs)

plt.legend(['normal accurancy', 'mutant accurancy'], loc='lower left')
plt.title('NEB mutation accurancy')
plt.xlabel('Mutation ratio')
plt.ylabel('Accurancy')
plt.show()


def NAI_experiment():
    normal_accs = []
    mutant_accs = []

    for mutation_ratio in mutation_ratios:
        trained_mutated_model = model_mut_opts.NAI_mut(trained_model, mutation_ratio)
        trained_mutated_model = network.compile_model(trained_mutated_model)

        loss, acc = trained_model.evaluate(test_datas, test_labels, verbose=False)
        normal_accs.append(acc)
        mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
        mutant_accs.append(mutant_acc)

    return normal_accs, mutant_accs


normal_accs, mutant_accs = NAI_experiment()

plt.axis([0, 1, 0, 1])
plt.plot(mutation_ratios, normal_accs)
plt.plot(mutation_ratios, mutant_accs)

plt.legend(['normal accurancy', 'mutant accurancy'], loc='lower left')
plt.title('NAI mutation accurancy')
plt.xlabel('Mutation ratio')
plt.ylabel('Accurancy')
plt.show()


def NS_experiment():
    normal_accs = []
    mutant_accs = []

    for mutation_ratio in mutation_ratios:
        trained_mutated_model = model_mut_opts.NS_mut(trained_model, mutation_ratio)
        trained_mutated_model = network.compile_model(trained_mutated_model)

        loss, acc = trained_model.evaluate(test_datas, test_labels, verbose=False)
        normal_accs.append(acc)
        mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
        mutant_accs.append(mutant_acc)

    return normal_accs, mutant_accs


normal_accs, mutant_accs = NS_experiment()

plt.axis([0, 1, 0, 1])
plt.plot(mutation_ratios, normal_accs)
plt.plot(mutation_ratios, mutant_accs)

plt.legend(['normal accurancy', 'mutant accurancy'], loc='lower left')
plt.title('NS mutation accurancy')
plt.xlabel('Mutation ratio')
plt.ylabel('Accurancy')
plt.show()
