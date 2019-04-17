from __future__ import print_function

import argparse

from .configs import bcolors
from .utils import *
import numpy as np

from .preprocessing import Preprocessing
from .triage_model import TriageModel

from keras import backend as K
from .neuron_coverage import NeuronCoverage

import time

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation')
parser.add_argument('transformation', help="realistic transformation type", choices=['synonym'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", default=1, type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", default=0.1, type=float)
parser.add_argument('step', help="step size of gradient descent", default=10, type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", default=0, type=float)
parser.add_argument('coverage', help="coverage option", choices=['dxp', 'kmnc', 'snbc'])
parser.add_argument('test_generation', help="test generation", choices=['dxp', 'basic', 'fgsm'])

parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)

args = parser.parse_args()

# =====================================================================================================================
# load 3 model and word2vec model

sum_vec = 1
word_dim = 300
# TB_SUMMARY_DIR = './onlycnn/add1/filter512/234/relu'
test_percent = 0.1

preprocessing = Preprocessing(word_dim)
assign_num = preprocessing.assign_num

train_raw_x, train_raw_y, train_vec_x, train_one_hot_y = preprocessing.generate_train_data(test_percent)
test_raw_x, test_raw_y, test_vec_x, test_one_hot_y = preprocessing.generate_test_data(test_percent)

# load multiple models sharing same input tensor
model1_model = TriageModel(assign_size=assign_num
                           , word_dim=word_dim
                           , max_word_size=int(500 / sum_vec)
                           , filter=512, model_id=1)
model2_model = TriageModel(assign_size=assign_num
                           , word_dim=word_dim
                           , max_word_size=int(500 / sum_vec)
                           , filter=512, model_id=2)
model3_model = TriageModel(assign_size=assign_num
                           , word_dim=word_dim
                           , max_word_size=int(500 / sum_vec)
                           , filter=512, model_id=3)

model1_model.train_learning()
model2_model.train_learning()
model3_model.train_learning()

model1 = model1_model.model
model2 = model2_model.model
model3 = model3_model.model

input_tensor = model1.input

neuron_coverage = NeuronCoverage()

# init neuron coverage table
model_layer_dict1, model_layer_dict2, model_layer_dict3 = neuron_coverage.init_coverage_tables(model1, model2, model3)
max_threshold_dict1, max_threshold_dict2, max_threshold_dict3 \
    , min_threshold_dict1, min_threshold_dict2, min_threshold_dict3 \
    = neuron_coverage.init_neuron_threshold_tables(model1, model2, model3, test_vec_x)

# =====================================================================================================================
# start generate inputs

start_time = time.time()
generated_bug_report = []
generated_bug_triage = []

i = 0
gen_index = 0
for _ in range(args.seeds):
    print(bcolors.HEADER + "seed %d" % i + bcolors.ENDC)
    i += 1

    # words at bug report to vector
    gen = test_raw_x[gen_index]
    gen_triage = test_raw_y[gen_index]
    gen_value = test_vec_x[gen_index]
    gen_value = np.expand_dims(gen_value, axis=0)
    orig = gen.copy()

    # first check if input already induces differences
    pred1, pred2, pred3 = model1.predict(gen_value), model2.predict(gen_value), model3.predict(gen_value)
    label1, label2, label3 = np.argmax(pred1[0]), np.argmax(pred2[0]), np.argmax(pred3[0])

    origin_label = np.argmax(test_one_hot_y[gen_index])

    if not label1 == label2 == label3 == origin_label:
        print(bcolors.OKGREEN + 'different target label and origin_label : {}, {}, {}, {}'.format(str(label1),
                                                                                                  str(label2),
                                                                                                  str(label3),
                                                                                                  str(origin_label))
              + bcolors.ENDC)

        gen_deprocessed = preprocessing.data_deprocess(gen)
        origin_deprocessed = preprocessing.data_deprocess(orig)

        generated_bug_report.append(gen_deprocessed)
        generated_bug_triage.append(gen_triage)
        gen_index += 1
        continue

    if not label1 == label2 == label3:
        print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(str(label1),
                                                                                            str(label2),
                                                                                            str(label3)) + bcolors.ENDC)

        neuron_coverage.update_coverage(gen_value, model1,
                                        model_layer_dict1, max_threshold_dict1, min_threshold_dict1,
                                        args.coverage, args.threshold)
        neuron_coverage.update_coverage(gen_value, model2,
                                        model_layer_dict2, max_threshold_dict2, min_threshold_dict2,
                                        args.coverage, args.threshold)
        neuron_coverage.update_coverage(gen_value, model3,
                                        model_layer_dict3, max_threshold_dict3, min_threshold_dict3,
                                        args.coverage, args.threshold)

        print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
              % (len(model_layer_dict1), neuron_coverage.neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                 neuron_coverage.neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                 neuron_coverage.neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
        averaged_nc = (neuron_coverage.neuron_covered(model_layer_dict1)[0] +
                       neuron_coverage.neuron_covered(model_layer_dict2)[0] +
                       neuron_coverage.neuron_covered(model_layer_dict3)[0]) / \
                      float(neuron_coverage.neuron_covered(model_layer_dict1)[1] +
                            neuron_coverage.neuron_covered(model_layer_dict2)[1] +
                            neuron_coverage.neuron_covered(model_layer_dict3)[1])
        print(bcolors.OKBLUE + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)
        max_nc = max(
            [neuron_coverage.neuron_covered(model_layer_dict1)[2], neuron_coverage.neuron_covered(model_layer_dict2)[2],
             neuron_coverage.neuron_covered(model_layer_dict3)[2]])
        print(bcolors.OKBLUE + 'max covered neurons %.3f' % max_nc + bcolors.ENDC)

        gen_deprocessed = preprocessing.data_deprocess(gen)
        origin_deprocessed = preprocessing.data_deprocess(orig)

        generated_bug_report.append(gen_deprocessed)
        generated_bug_triage.append(gen_triage)
        gen_index += 1

        # save the result to disk
        f = open('./generated_inputs/' + "already_differ_" + args.transformation + "_"
                 + args.coverage + "_" + args.test_generation + "_"
                 + str(label1) + "_" + str(label2)
                 + "_" + str(label3) + ".txt", 'w')
        f.write(gen_deprocessed)
        f.close()

        f = open('./generated_inputs/' + "origin_" + args.transformation + "_"
                 + args.coverage + "_" + args.test_generation + "_"
                 + str(label1) + "_" + str(label2)
                 + "_" + str(label3) + ".txt", 'w')
        f.write(origin_deprocessed)
        f.close()
        continue

    # if all label agrees
    orig_label = label1
    layer_name1, index1 = neuron_coverage.neuron_to_cover(model_layer_dict1)
    layer_name2, index2 = neuron_coverage.neuron_to_cover(model_layer_dict2)
    layer_name3, index3 = neuron_coverage.neuron_to_cover(model_layer_dict3)

    # construct joint loss function
    if args.target_model == 0:
        loss1 = -args.weight_diff * K.mean(model1.get_layer('predictions').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('predictions').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('predictions').output[..., orig_label])
    elif args.target_model == 1:
        loss1 = K.mean(model1.get_layer('predictions').output[..., orig_label])
        loss2 = -args.weight_diff * K.mean(model2.get_layer('predictions').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('predictions').output[..., orig_label])
    elif args.target_model == 2:
        loss1 = K.mean(model1.get_layer('predictions').output[..., label1])
        loss2 = K.mean(model2.get_layer('predictions').output[..., orig_label])
        loss3 = -args.weight_diff * K.mean(model3.get_layer('predictions').output[..., orig_label])

    loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
    loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])
    loss3_neuron = K.mean(model3.get_layer(layer_name3).output[..., index3])
    layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (loss1_neuron + loss2_neuron + loss3_neuron)

    # for adversarial bug report generation
    final_loss = K.mean(layer_output)

    # we compute the gradient of the bug report wrt this loss
    grads = neuron_coverage.normalize(K.gradients(final_loss, input_tensor)[0])

    # this function returns the loss and grads given the bug report
    iterate = K.function([input_tensor], [grads])

    # we run gradient ascent for grad_iterations
    for iters in range(args.grad_iterations):
        grads_value = iterate([gen_value])

        # =============================================================================================================
        # start to adversarial bug report generation
        if args.transformation == 'synonym':
            gen, return_type = constraint_synonym(gen, grads_value, args.test_generation,
                                                  preprocessing.word_vec_modeler)

        if not return_type:
            continue

        # generated bug report to vector
        test_raw_x[gen_index] = gen
        gen = [gen]
        temp_vec_x, gen_value = preprocessing.data_preprocess(gen)
        # gen_value = temp_vec_x[gen_index]
        # gen_value = np.expand_dims(gen_value, axis=0)

        # =============================================================================================================

        pred1, pred2, pred3 = model1.predict(gen_value), model2.predict(gen_value), model3.predict(gen_value)
        label1, label2, label3 = np.argmax(pred1[0]), np.argmax(pred2[0]), np.argmax(pred3[0])

        gen = gen[0]

        print(bcolors.UNDERLINE + "iters %d" % (iters + 1) + bcolors.ENDC)
        print("label1 :", label1, " label2 :", label2, " label3 :", label3)
        if not label1 == label2 == label3:
            neuron_coverage.update_coverage(gen_value, model1, model_layer_dict1, max_threshold_dict1,
                                            min_threshold_dict1,
                                            args.coverage)
            neuron_coverage.update_coverage(gen_value, model2, model_layer_dict2, max_threshold_dict2,
                                            min_threshold_dict2,
                                            args.coverage)
            neuron_coverage.update_coverage(gen_value, model3, model_layer_dict3, max_threshold_dict3,
                                            min_threshold_dict3,
                                            args.coverage)

            print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (
                  len(model_layer_dict1), neuron_coverage.neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                  neuron_coverage.neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                  neuron_coverage.neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
            averaged_nc = (neuron_coverage.neuron_covered(model_layer_dict1)[0] +
                           neuron_coverage.neuron_covered(model_layer_dict2)[0] +
                           neuron_coverage.neuron_covered(model_layer_dict3)[0]) / \
                          float(neuron_coverage.neuron_covered(model_layer_dict1)[1] +
                                neuron_coverage.neuron_covered(model_layer_dict2)[1] +
                                neuron_coverage.neuron_covered(model_layer_dict3)[1])
            print(bcolors.OKBLUE + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)
            max_nc = max([neuron_coverage.neuron_covered(model_layer_dict1)[2],
                          neuron_coverage.neuron_covered(model_layer_dict2)[2],
                          neuron_coverage.neuron_covered(model_layer_dict3)[2]])
            print(bcolors.OKBLUE + 'max covered neurons %.3f' % max_nc + bcolors.ENDC)

            gen_deprocessed = preprocessing.data_deprocess(gen)
            orig_deprocessed = preprocessing.data_deprocess(orig)

            generated_bug_report.append(gen_deprocessed)
            generated_bug_triage.append(gen_triage)

            #
            # # save the result to disk
            f = open('./generated_inputs/' + args.transformation + "_"
                     + args.coverage + "_" + args.test_generation + "_" + str(label1) + "_"
                     + str(label2) + "_" + str(label3) + ".txt", 'w')
            f.write(gen_deprocessed)
            f.close()

            f = open('./generated_inputs/' + "origin_" + args.transformation + "_"
                     + args.coverage + "_" + args.test_generation + "_"
                     + str(label1) + "_" + str(label2)
                     + "_" + str(label3) + ".txt", 'w')
            f.write(orig_deprocessed)
            f.close()
            break

    gen_index += 1

generated_all_saved_file = open('generated_test_data.txt', 'a')
for index, bug_report in enumerate(generated_bug_report):
    generated_all_saved_file.write(bug_report)
    generated_all_saved_file.write('\n')
generated_all_saved_file.close()

generated_all_triage_saved_file = open('generated_assignTo.txt', 'a')
for index, triage in enumerate(generated_bug_triage):
    generated_all_triage_saved_file.write(triage)
    generated_all_triage_saved_file.write('\n')
generated_all_triage_saved_file.close()

print("----- %s seconds -----" % (time.time() - start_time))
