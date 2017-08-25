

import os
import csv
import itertools
import nltk
unknown_token = "UNKNOWN_CODE"
sentence_start_token = "CODE_START"
sentence_end_token = "CODE_END"
encoding = 'utf-8'
data_dir = "data/tinyshakespeare"
input_file = os.path.join(data_dir, "allJavaSourceCode.txt")
with open(input_file, 'r', encoding=encoding) as f:
    reader = f.readlines()

    # Split full comments into sentences
    sentences = []
    for x in reader:
        sentences.append(sentence_start_token + " " + x + " " + sentence_end_token)

tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))
# vocab = word_freq.most_common(vocabulary_size)
# Get the most common words and build index_to_word and word_to_index vectors
index_to_word = [x for x in word_freq]
index_to_word.append(unknown_token)

word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]


chars = index_to_word  ##동일한 자료형으로 묶어 준다. ex) 현재 count_paris = 00-> ('',145563), 01->('a',154343425) zip실행하면 ['','a',,,,,],[142345, 1534232424]이런
vocab_size = len(chars)
vocab = word_to_index
