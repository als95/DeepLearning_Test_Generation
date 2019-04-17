import random as r
import numpy as np

from nltk.corpus import wordnet

from sklearn.metrics.pairwise import cosine_similarity


# ====================================================================================
#                           test data generate process
# ====================================================================================

# generate synonym of random word in bug report
# Input
# : gen - random choose bug report, grads_value - grads given gen(bug report)
#   , test_generation_name - test generation method, word_vec_modeler - word vector model
# Output
# : gen - transformed gen, return_type - success or fail transform
def constraint_synonym(gen, grads_value, test_generation_name, word_vec_modeler):
    syns_list = []
    k = 0
    j = 0
    i = r.randint(0, len(gen) - 1)
    return_type = False

    if test_generation_name == 'basic':
        iterate = np.amax(grads_value)

        while j < iterate and i < len(gen):
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
                max_similarity_value = 0
                max_similarity_index = 0

                for syns_index, _ in enumerate(syns_list):
                    syns = syns_list[syns_index].lemmas()[0].name()
                    if syns != gen[i]:
                        similarity = similarity_between_words(gen[i], syns, word_vec_modeler)
                        if similarity is False:
                            continue
                        similarity_value = similarity[0][0]
                        if similarity_value < 0.5:
                            continue

                        if similarity_value > max_similarity_value:
                            max_similarity_value = similarity_value
                            max_similarity_index = syns_index

                if max_similarity_value != 0 and max_similarity_value > 0.5:
                    return_type = True
                    gen[i] = syns_list[max_similarity_index].lemmas()[0].name()

            syns_list = []
            j += 1
            i = r.randint(0, len(gen) - 1)

        return gen, return_type

    elif test_generation_name == 'dxp':
        iterate = np.mean(grads_value)

        while j < iterate and k < len(gen):
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
                max_similarity_value = 0
                max_similarity_index = 0

                for syns_index, _ in enumerate(syns_list):
                    syns = syns_list[syns_index].lemmas()[0].name()
                    if syns != gen[i]:
                        similarity = similarity_between_words(gen[i], syns, word_vec_modeler)
                        if similarity is False:
                            continue
                        similarity_value = similarity[0][0]
                        if similarity_value < 0.5:
                            continue

                        if similarity_value > max_similarity_value:
                            max_similarity_value = similarity_value
                            max_similarity_index = syns_index

                if max_similarity_value != 0 and max_similarity_value > 0.5:
                    return_type = True
                    gen[i] = syns_list[max_similarity_index].lemmas()[0].name()

            syns_list = []
            i = r.randint(0, len(gen) - 1)
            j += 1
            k += 1

        return gen, return_type

    elif test_generation_name == 'fgsm':
        grad_mean = np.mean(grads_value)
        iterate = np.sign(grad_mean)

        while j < iterate and k < len(gen):
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
                max_similarity_value = 0
                max_similarity_index = 0

                for syns_index, _ in enumerate(syns_list):
                    syns = syns_list[syns_index].lemmas()[0].name()
                    if syns != gen[i]:
                        similarity = similarity_between_words(gen[i], syns, word_vec_modeler)
                        if similarity is False:
                            continue
                        similarity_value = similarity[0][0]
                        if similarity_value < 0.5:
                            continue

                        if similarity_value > max_similarity_value:
                            max_similarity_value = similarity_value
                            max_similarity_index = syns_index

                if max_similarity_value != 0 and max_similarity_value > 0.5:
                    return_type = True
                    gen[i] = syns_list[max_similarity_index].lemmas()[0].name()

            syns_list = []
            i = r.randint(0, len(gen) - 1)
            j += 1
            k += 1

        return gen, return_type


# calculate similarity between generated word and synonym
# Input
# : gen - a word of bug report, syns - synonym of gen, word_vec_modeler - word2vec model
# Output
# : similarity between generated word and synonym
def similarity_between_words(gen, syns, word_vec_modeler):
    gen_value = word_vec_modeler.get_vector_from_word(gen)
    syns_value = word_vec_modeler.get_vector_from_word(syns)

    if syns_value is None or gen_value is None:
        return False

    gen_value.flatten()
    syns_value.flatten()

    gen_value = np.expand_dims(gen_value, axis=0)
    syns_value = np.expand_dims(syns_value, axis=0)

    return cosine_similarity(gen_value, syns_value)
