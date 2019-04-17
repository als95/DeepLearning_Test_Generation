import os

import utils
import source_mut_operators
import network_triage
import tensorflow as tf

import numpy as np

class SourceMutateScore:

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.utils = utils.GeneralUtils()

        self.network = network_triage.TriageNetwork()

        self.error_by_classes =  np.zeros(17)
    

    def integration_test(self):
        # modes = ['DR', 'LE', 'DM', 'DF', 'NP', 'LR', 'LAs', 'AFRs']
        modes = ['DR', 'DM', 'DF', 'NP', 'LR', 'LAs', 'AFRs']

        # Model creation
        # This should variates according to the value of self.model_architecture
        train_dataset, test_dataset = self.network.load_data()
        generated_dataset = self.network.load_generated_data()

        num_mutant_error_rates = 0

        num_mutant_killed_classes_of_train = 0
        num_mutant_killed_classes_of_test = 0
        num_mutant_killed_classes_of_generated = 0

        error_rate_by_models = []

        train_data, train_label = train_dataset
        test_data, test_label = test_dataset
        generated_data, generated_label = generated_dataset
        #
        # temp_data = np.concatenate((train_data, test_data), axis=0)
        # temp_label = np.concatenate((train_label, test_label), axis=0)

        # generated_data = np.concatenate((test_data, generated_data), axis=0)
        # generated_label = np.concatenate((test_label, generated_label), axis=0)

        generated_dataset = (generated_data, generated_label)

        # Test for generate_model_by_source_mutation function
        for index, mode in enumerate(modes):
            file_name = mode + '_model'
            model = self.network.load_model(file_name)

            # train data
            num_mutant_killed_class, _ = self.cal_mutate_score_by_source_mutation(train_dataset, model, mode)
            num_mutant_killed_classes_of_train += num_mutant_killed_class
	
            # test data
            num_mutant_killed_class, num_mutant_error_rate = self.cal_mutate_score_by_source_mutation(test_dataset,
                                                                                                      model, mode)
            num_mutant_killed_classes_of_test += num_mutant_killed_class

            # generated data
            # num_mutant_killed_class, _ = self.cal_mutate_score_by_source_mutation(generated_dataset, model, mode)
            # num_mutant_killed_classes_of_generated += num_mutant_killed_class
            num_mutant_error_rates += num_mutant_error_rate

            error_rate_by_models.append(num_mutant_error_rate)
            del model
	
        mutant_score_of_train = num_mutant_killed_classes_of_train / (len(modes) * 16)
        mutant_score_of_test = num_mutant_killed_classes_of_test / (len(modes) * 16)
        mutant_score_of_generated = num_mutant_killed_classes_of_generated / (len(modes) * 16)
        error_rate = num_mutant_error_rates / len(modes)

        print("Mutant Score of train data : ", mutant_score_of_train * 100)
        print("Mutant Score of test data : ", mutant_score_of_test * 100)
        print("Mutant Score of generated data : ", mutant_score_of_generated * 100)
        print("Average Error rate of this test data : ", error_rate * 100)

        print("Error rate")
        for index, rate in enumerate(error_rate_by_models):
            print(modes[index], " : ", rate)

        print("Error Distribution")
        print(self.error_by_classes)


    def cal_mutate_score_by_source_mutation(self, test_dataset, model, mode):
        valid_modes = ['DR', 'LE', 'DM', 'DF', 'NP', 'LR', 'LAs', 'AFRs']
        assert mode in valid_modes, 'Input mode ' + mode + ' is not implemented'

        test_datas, test_labels = test_dataset

        # np.concatenate((test_datas, self.generated_datas), axis=None)
        # test_datas.extend(self.generated_datas)
        # np.concatenate((test_labels, self.generated_labels), axis=None)
        # test_labels.extend(self.generated_labels)

        self.network.evaluate_model(model, test_datas, test_labels, mode)
        return self.evaluate_mut_model(model, test_datas, test_labels)

    def evaluate_mut_model(self, mut_model, test_datas, test_labels):
        killed_classes = np.zeros(17, dtype=bool)

        num_error_cases = 0

        for index, test_data in enumerate(test_datas):
            test_data = np.expand_dims(test_data, axis=0)
            predict = mut_model.predict(test_data)
            predict_label = np.argmax(predict[0])
            result_label = np.argmax(test_labels[index])

            if int(result_label) != int(predict_label):
                num_error_cases += 1
                killed_classes[int(result_label)] = True
                self.error_by_classes[int(result_label)] += 1

        num_killed_classes = 0
        for killed_class in killed_classes:
            if killed_class:
                num_killed_classes += 1

        error_rate = num_error_cases / len(test_labels)

        return num_killed_classes, error_rate

if __name__ == "__main__":

    source_mut_score = SourceMutateScore()
    source_mut_score.integration_test()