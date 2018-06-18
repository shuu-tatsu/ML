#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy.random import *
import numpy as np
import math
import logging
import random


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
        self.neg_label = [(line, 0) for line in neg_lines]

    def merge(self):
        self.mixed_dataset = self.pos_label + self.neg_label
        return self.mixed_dataset


def load(dataset):
    train_set = Dataset(dataset[0], dataset[1])
    train_set.labeling()
    train_set.merge()
    random.shuffle(train_set.mixed_dataset)
    xs_train = [i[0] for i in train_set.mixed_dataset]
    ys_train = [i[1] for i in train_set.mixed_dataset]
    np.save('xs_train.npy', xs_train)
    np.save('ys_train.npy', ys_train)


def main(train_set, test_set):
    load(train_set)


if __name__ == '__main__':
    logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    """
    train_pos_set = 'positive.review'
    train_neg_set = 'negative.review'
    """
    train_pos_set = 'pos_train.review'
    train_neg_set = 'neg_train.review'
    test_pos_set = 'pos_test.review'
    test_neg_set = 'neg_test.review'

    train_set = [train_pos_set, train_neg_set]
    test_set =  [test_pos_set, test_neg_set]
    main(train_set, test_set)
