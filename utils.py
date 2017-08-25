import codecs
import os
import collections
import csv
import itertools
import nltk
from six.moves import cPickle
import numpy as np
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "CODE_START"
sentence_end_token = "CODE_END"
class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size #50
        self.seq_length = seq_length #50
        self.encoding = encoding # UTF-8

        input_file = os.path.join(data_dir, "allJavaSourceCode.csv")## os.path.join(dir, string) = dir+string
        vocab_file = os.path.join(data_dir, "vocab.pkl")## input.txt에서 가져온 char들의 유일한 사전 데이터 단어와 사전의 index 번호 튜플 지정
        tensor_file = os.path.join(data_dir, "data.npy")## tensor정보 저장

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)): ##이미 만들어 놓은 model data 있는 경우 proprocess진행
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else: ##없는 경우 기존에 있던 모델을 가져온다
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with open(input_file, 'r',encoding=self.encoding) as f:
            reader = f.readlines()

            # Split full comments into sentences
            sentences = []
            for x in reader:
                sentences.append(sentence_start_token + " " + x + " " + sentence_end_token)

        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("Found %d unique words tokens." % len(word_freq.items()))

        # Get the most common words and build index_to_word and word_to_index vectors
        index_to_word = [x for x in word_freq]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

        self.chars = index_to_word  ##동일한 자료형으로 묶어 준다. ex) 현재 count_paris = 00-> ('',145563), 01->('a',154343425) zip실행하면 ['','a',,,,,],[142345, 1534232424]이런
        self.vocab_size = len(self.chars)
        self.vocab = word_to_index

        x_data =[]
        for sent in tokenized_sentences:
            for w in sent:
                x_data.append(word_to_index[w])

        data = np.array(x_data)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)  ## 사전에 유일한 charactor들을 저장
        self.tensor = np.array(data)  ## map 을 이용해서 함수에 대응되는 data의 값을 하나씩 대입하여 대응되는 int수를 array화 시킨다
        np.save(tensor_file, self.tensor)  ## 이 정보를 data.npy에 저장


    def load_preprocessed(self, vocab_file, tensor_file):##저장된 모델이 있는 경우
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f) ## 저장된 Char를 불러온다.
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length)) ##제거해도 될듯 실험 해봐야함

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."##assert -> 다른 언어로 치면exception

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]## x_data의 첫번째 값을 Y_data의 마지막 값에 대입
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1) #split(A,N(N개의 동일한 숫자),1=x축 or 0=y축 기준으로)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
