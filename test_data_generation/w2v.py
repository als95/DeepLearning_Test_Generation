import codecs

import gensim as gensim
from nltk.corpus import stopwords
import re

from sklearn.feature_extraction.text import CountVectorizer


class WordVecModel:
    def __init__(self, dim):
        self.dim = dim
        self.model = None
        self.doc_x = None
        pass

    def train_word_vec(self, input_vector, model_name, min_count=5, window=5, epoch=5, sg=1):
        self.model = gensim.models.Word2Vec(input_vector, min_count=min_count, size=self.dim, window=window,
                                            sg=sg, iter=epoch, workers=8)
        self.model.save(model_name)
        return self.model

    def load_word_vec(self, model_name):
        # self.model = gensim.models.Word2Vec.load(model_name)
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=True)
        return self.model

    def get_vector_from_word(self, word):
        if self.model is None:
            raise ModuleNotFoundError("불러와진 모델이 없습니다. load_word_vec 을 먼저 호출해주세요")
        if word in self.model:
            return self.model[word]
        else:
            return None

    def train_doc_vec(self, input_vector,doc_dim):
        vectorizer = CountVectorizer(analyzer="word",
                                     tokenizer=None,
                                     token_pattern=r"(?u)\b\S\S+\b",
                                     preprocessor=None,
                                     stop_words=None,
                                     max_features=doc_dim)
        fit_data = []
        for i in range(0, len(input_vector)):
            fit_string = ""
            for j in range(0, len(input_vector[i])):
                fit_string = (fit_string + " " + str(input_vector[i][j]))
            fit_data.append(fit_string)
        train_data_features = vectorizer.fit_transform(fit_data)
        self.doc_x = train_data_features.toarray()
        return self.doc_x


class WordPreprocessor:
    def data_preprocessing(self, input_data_path, stop_words='english',
                           remove_only_number=True, remove_sepcial_symbol=True, split_with_dot=True,
                           split_camel_case=True):
        input_file = codecs.open(input_data_path, "r", "utf-8")

        # sarr은 desc를 정제한 배열
        sarr = [[]]

        # stopword를 거르고 따로 독립적으로 존재하는 특수기호들도 거른다.
        stop_words = set(stopwords.words(stop_words))
        stop_words.update(['-', '.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

        sarr.pop(0)
        # 파일을 라인바이 라인으로 읽으면서 스탑워드에 해당되는 항목을 배열에서 배제 시킨다.
        for line in input_file:
            line = line.strip()
            line = ' '.join([word for word in line.split() if word not in stop_words])
            temp = line.split()
            sarr.append(temp)

        # 여기서 특수기호를 거르고 camel 케이스를 공백으로 고침
        for i in range(0, len(sarr)):
            for j in range(0, len(sarr[i])):
                # 특수기호 자체로만 존재하는 배열 요소들을 빈 스트링으로 치환
                if remove_sepcial_symbol is True:
                    sarr[i][j] = re.sub("[-^~#@\">$<|{}\[\](;*&:?!=/),_'+]", '', sarr[i][j])
                # 카멜케이스인 단어를 소문자 단어 공백 소문자 단어로 바꿈
                if split_camel_case is True:
                    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', sarr[i][j])
                    sarr[i][j] = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()

        # .이 여러번 나오는 것은 보통 소스코드 .을 공백으로 치환한다.
        for i in range(0, len(sarr)):
            for j in range(0, len(sarr[i])):
                if ''.join(sarr[i][j]).count('.') > 0:
                    if split_with_dot is True:
                        sarr[i][j] = ''.join(sarr[i][j]).replace('.', ' ')

        # 공백이 있는 단어들은 공백을 기준으로 스플릿한다음 원래 배열에 추가한다.
        for i in range(0, len(sarr)):
            for j in range(0, len(sarr[i])):
                if ''.join(sarr[i][j]).count(' ') > 0:
                    temp = ''.join(sarr[i][j]).split()
                    for k in range(0, len(temp)):
                        sarr[i].append(temp[k])

        # 공백 추가한 단어는 이제 필요없으므로 삭제한다.
        for i in range(0, len(sarr)):
            n = len(sarr[i])
            j = 0
            while j < n:
                # 만약 공백이 하나 이상 있다면 그 원소를 삭제하고 n의 카운트를 감소시킨다.
                if ''.join(sarr[i][j]).count(' ') > 0:
                    sarr[i].pop(j)
                    n = n - 1
                # 공백이 없다면 그대로 j값을 증가시킴
                else:
                    j = j + 1

        # .삭제 및 순수 숫자로 구성된 단어 삭제
        for i in range(0, len(sarr)):
            for j in range(0, len(sarr[i])):
                # 순수 숫자로 구성된 단어를 빈 스트링으로 치환한다.
                if remove_only_number is True:
                    sarr[i][j] = re.sub('^[0-9]+$', '', sarr[i][j])
                # .이 붙어있거나 .이 포함된 단어를 빈스트링으로 치환
                sarr[i][j] = ''.join(sarr[i][j]).replace('.', '')

        # 여태껏 빈스트링으로 치환한 정보는 배열안에 남아 있을 필요가 없기 때문에 전부 제거한다.
        for i in range(0, len(sarr)):
            while '' in sarr[i]:
                sarr[i].remove('')

        return sarr
