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
    

    def integration_test(self, verbose=False):
        modes = ['DR', 'LE', 'DM', 'DF', 'NP', 'LR', 'LAs', 'AFRs']

        # Model creation
        # This should variates according to the value of self.model_architecture
        train_dataset, test_dataset = self.network.load_data()

        num_mutant_killed_classes = 0

        # Test for generate_model_by_source_mutation function 
        for index, mode in enumerate(modes):
            file_name = mode + '_model'
            model = self.network.load_model(file_name)
            num_mutant_killed_classes += self.cal_mutate_score_by_source_mutation(train_dataset, test_dataset, model,
                                                                                  mode, verbose=verbose)
            del model

        mutant_score = num_mutant_killed_classes / (len(modes) * 17)
        print("Mutant Score of this test data : ", mutant_score * 100)


    def cal_mutate_score_by_source_mutation(self, train_dataset, test_dataset, model, mode, verbose=False):
        valid_modes = ['DR', 'LE', 'DM', 'DF', 'NP', 'LR', 'LAs', 'AFRs']
        assert mode in valid_modes, 'Input mode ' + mode + ' is not implemented'

        test_datas, test_labels = test_dataset

        self.network.evaluate_model(model, test_datas, test_labels, mode)
        return self.evaluate_mut_model(model, test_datas, test_labels)

    def evaluate_mut_model(self, mut_model, test_datas, test_labels):
        killed_classes = np.zeros(17, dtype=bool)
        for index, test_data in enumerate(test_datas):
            test_data = np.expand_dims(test_data, axis=0)
            predict = mut_model.predict(test_data)
            predict_label = np.argmax(predict[0])
            result_label = np.argmax(test_labels[index])

            if int(result_label) != int(predict_label):
                killed_classes[int(result_label)] = True

        num_killed_classes = 0
        for killed_class in killed_classes:
            if killed_class:
                num_killed_classes += 1

        return num_killed_classes

if __name__ == "__main__":

    source_mut_score = SourceMutateScore()
    source_mut_score.integration_test()