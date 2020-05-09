import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import HTML
from itertools import chain
from collections import Counter, defaultdict
from helpers import show_model, Dataset
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution

data = Dataset("tags-universal.txt", "brown-universal.txt", train_test_split=0.8)

print("There are {} sentences in the corpus.".format(len(data)))
print("There are {} sentences in the training set.".format(len(data.training_set)))
print("There are {} sentences in the testing set.".format(len(data.testing_set)))

assert len(data) == len(data.training_set) + len(data.testing_set), \
    "The number of sentences in the training set + testing set should sum to the number of sentences in the corpus"

key = 'b100-38532'
print("Sentence: {}".format(key))
print("words:\t{!s}".format(data.sentences[key].words))
print("tags:\t{!s}".format(data.sentences[key].tags))

print("There are a total of {} samples of {} unique words in the copus."
      .format(data.N, len(data.vocab)))
print("There are {} samples of {} unique words in the training set."
      .format(data.training_set.N, len(data.training_set.vocab)))
print("There are {} samples of {} unique words in the testing set."
      .format(data.testing_set.N, len(data.testing_set.vocab)))
print("There are {} words in the test set that are missing in the training set."
      .format(len(data.testing_set.vocab - data.training_set.vocab)))

assert data.N == data.training_set.N + data.testing_set.N, \
    "The number of training + test samples should sum to the total number of samples"

for i in range(2):
    print("sequence {}:".format(i + 1), data.X[i])  # words
    print("Labels {}:".format(i + 1), data.Y[i])  # tags

print("\nStream (word, tag) pairs:\n")
for i, pair in enumerate(data.stream()):
    print("\t", pair)
    if i > 5:
        break
tags = (tag for i, (word, tag) in enumerate(data.training_set.stream()))
words = (word for i, (word, tag) in enumerate(data.training_set.stream()))
tags_words = defaultdict(list)
words_dic = defaultdict(list)


def pair_counts(sequence_A=None, sequences_B=None):
    for i, (tag, word) in enumerate(zip(sequence_A, sequences_B)):
        tags_words[tag].append(word)
    for k in tags_words.keys():
        words_dic[k] = Counter(tags_words[k])
    return words_dic


emission_counts = pair_counts(tags, words)

word_tag_dic = defaultdict(list)
most_frequent = defaultdict(list)

from collections import namedtuple

FakeState = namedtuple("FakeState", "name")


class MFCTagger:
    missing = FakeState(name="<MISSING>")

    def __init__(self, table):
        self.table = defaultdict(lambda: MFCTagger.missing)
        self.table.update({word: FakeState(name=tag) for word, tag in table.items()})
    def viterbi(self, seq):
        return 0., list(enumerate(["<start>"] + [self.table[w] for w in seq] + ["<end>"]))

for i, (tag, word) in enumerate(zip(tags, words)):
    word_tag_dic[word].append(tag)
for k in word_tag_dic.keys():
    most_frequent[k] = (Counter(word_tag_dic[k])).most_common(1)[0][0]

mfc_table = most_frequent

mfc_model = MFCTagger(mfc_table)

#assert len(mfc_table) == len(data.training_set.vocab), ""
#assert all(k in data.training_set.vocab for k in mfc_table.keys()), ""
#assert sum(int(k not in mfc_table) for k in data.training_set.vocab) == 5521, ""

def replace_unknown(sequence):
    return [w if w in data.training_set.vocab else 'nan' for w in sequence]

def simplify_decoding(X, model):
    _, state_path = model.viterbi(replace_unknown(X))
    return [state[1].name for state in state_path[1: -1]]

for key in data.testing_set.keys[:3]:
    print("Sentence Key: {}\n".format(key))
    print("Predicted labels:\n-----------------")
    print(simplify_decoding(data.sentences[key].words, mfc_model))
    print()
    print("Actual labels:\n--------------")
    print(data.sentences[key].tags)
    print("\n")

tags = [tag for i, (tag, word) in enumerate(data.training_set.stream())]
def unigram_counts(sequences):
    tag_counter = {}
    for tag in sequences:
        if tag in tag_counter.keys():
            tag_counter[tag] +=1
        else:
            tag_counter[tag] = 1
    return tag_counter

tag_unigrams = unigram_counts(tags)

tags = [tag for i, (word, tag) in enumerate(data.training_set.stream())]
tag_pairs = list(zip(tags[:-1], tags[1:]))
tag_pair_count = {}

def bigram_counts(sequences):
    for tag_pair in sequences:
        if tag_pair in tag_pair_count.keys():
            tag_pair_count[tag_pair] +=1
        else:
            tag_pair_count[tag_pair] = 1
    return tag_pair_count

tag_bigrams = bigram_counts(tag_pairs)

starting_tags = defaultdict(list)

def starting_counts(sequences):
    for tag in data.training_set.tagset:
        starting_tags[tag] = len(seq[0] for seq in sequences if seq[0] == tag)
    return starting_tags

tag_starts = starting_counts(data.training_set.Y)

ending_tags = defaultdict(list)
def ending_counts(sequences):
    for tag in data.training_set.tagset:
        ending_tags[tag] = len([seq[-1] for seq in sequences if seq[-1] == tag])
    return ending_tags

tag_ends = ending_counts(data.training_set.Y)

basic_model = HiddenMarkovModel(name="base-hmm-tagger")
tag_word_counter = pair_counts(tags, words)
states = {}

for tag, words in tag_word_counter.items():
    words_tag_state = defaultdict(list)
    for word in words.keys():
        words_tag_state[word] = tag_word_counter[tag][word]/ tag_unigrams[tag]
    emission = DiscreteDistribution(dict(words_tag_state))
    states[tag] = State(emission, name = tag)

basic_model.add_states(list(states.values()))

for tag in data.training_set.tagset:
    state = states[tag]
    basic_model.add_transition(basic_model.start, state, tag_starts[tag]/len(data.training_set))

for tag in data.training_set.tagset:
    state= states[tag]
    basic_model.add_transition(state, basic_model.end, tag_ends[tag]/tag_unigrams[tag])

for tag1 in data.training_set.tagset:
    state1 = states[tag1]
    for tag2 in data.training_set.tagset:
        state2 = states[tag2]
        basic_model.add_transition(state1, state2, tag_bigrams[(tag1, tag2)]/tag_unigrams[tag1])

basic_model.bake()