#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class Dataset(object):
    def __init__(self, pos, neg):
        self.pos_text = pos
        self.neg_text = neg
        self.pos_label = []
        self.neg_label = []
        self.mixed_dataset = []

    def onehot_vec(self, word_number):
        #initialize word_vec
        word_vec = [0 for _ in range(200000)]
        word_vec[int(word_number)] = 1
        return word_vec

    def vectorizer(self, line_include_freq_info):
        list_word_vec = [self.onehot_vec(i.split(':')[0]) for i in line_include_freq_info]
        vec_line = np.sum(list_word_vec, axis=0)
        return vec_line

    def load(self, dataset):
        with open(dataset, 'r') as r:
            lines = [i for i in r.read().split('\n')]
            lines = [i.split() for i in lines[:-1]]
            vec_lines = [self.vectorizer(i) for i in lines]
            return vec_lines

    def labeling(self):
        pos_lines = self.load(self.pos_text)
        self.pos_label = [(line, 1) for line in pos_lines]
        neg_lines = self.load(self.neg_text)
        self.neg_label = [(line, -1) for line in neg_lines]

    def merge(self):
        self.mixed_dataset = self.pos_label + self.neg_label
        print(self.mixed_dataset)
        return self.mixed_dataset

class LR(object):
    # __init__: copy from zhao xin
    #def __init__(self, train_set, test_set, sizes, learning_rate, epoachs, minibatch_size):
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
        #self.learning_rate = learning_rate
        #self.epoachs = epoachs
        #self.minibatch_size = minibatch_size
        #self.w_ = np.zeros(size)
        #self.b_ = np.random.randn()
    
if __name__ == '__main__':
    train_set = Dataset('pos_train.review', 'neg_train.review')
    train_set.labeling()
    train_set.merge()

    test_set = Dataset('pos_test.review', 'neg_test.review')
    test_set.labeling()
    test_set.merge()

    lr_model = LR(train_set.mixed_dataset, test_set.mixed_dataset)

