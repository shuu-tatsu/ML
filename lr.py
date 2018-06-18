#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy.random import *
import numpy as np
import math


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
    xs_train = [i[0] for i in train_set.mixed_dataset]
    ys_train = [i[1] for i in train_set.mixed_dataset]
    return xs_train, ys_train


class GaussianInitializer(object):
    def __init__(self):
        self.dim = 0

    def apply(self, w):
        self.dim = w.shape[0]
        w = randn(self.dim)


def sigmoid(x):
    return math.exp(x) / (1 + math.exp(x))


class LogisticRegression(object):

    def __init__(self, w):
        self.w = w

    def grad(self, xs_train, ys_train):
        target = [0 for _ in range(200000)]
        for i in range(len(xs_train)):
            target = target + (xs_train[i] * (sigmoid(np.dot((self.w).T, xs_train[i]) - ys_train[i])))
        return target

    def forward(self, xs_train):
        ys_predict = [0 for _ in range(len(xs_train))]
        for i in range(len(xs_train)):
            ys_predict[i] = (sigmoid(np.dot((self.w).T, xs_train[i])))
        return ys_predict


class SGD(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def cross_entropy_error_func(self, model, xs_train, ys_train):
        model.w = model.w - self.learning_rate * model.grad(xs_train, ys_train)


def culc_accuracy(ys_predict, ys_train):
    ys_predict_label = [0 for _ in range(len(ys_predict))]
    for i in range(len(ys_predict)):
        if ys_predict[i] > 0.5:
            ys_predict_label[i] = 1
        else:
            ys_predict_label[i] = 0
    count = 0
    for i in range(len(ys_predict_label)):
        if ys_predict_label == ys_train:
            count += 1
    accuracy = count / len(ys_predict_label)
    return accuracy


def culc_loss(ys_predict, ys_train):
    loss_sum = 0
    for i in range(len(ys_predict)):
        loss_sum = loss_sum + abs(ys_predict[i] - ys_train[i])
    return loss_sum


def train(train_set, epochs, learning_rate):
    xs_train, ys_train = load(train_set)
    dim = xs_train[0].shape[0]
    w = np.empty((dim,), dtype=np.float16)
    initializer = GaussianInitializer()
    initializer.apply(w)
    model = LogisticRegression(w)
    optimizer = SGD(learning_rate)

    def process(xs_train, ys_train):
        ys_predict = model.forward(xs_train)
        loss = culc_loss(ys_predict, ys_train)
        accuracy = culc_accuracy(ys_predict, ys_train)
        optimizer.cross_entropy_error_func(model, xs_train, ys_train)
        return loss, accuracy

    for epoch in range(1, epochs + 1):
        loss, accuracy = process(xs_train, ys_train)

        print(
            "[{}] epoch {} - #samples: {}, loss: {:.8f}, accuracy: {:.8f}"
            .format("train", epoch, len(ys_train), loss, accuracy))

def test(test_set):
    test_set = None


def main(train_set, test_set, epochs, learning_rate):
    train(train_set, epochs, learning_rate)
    test(test_set)


if __name__ == '__main__':
    """
    train_pos_set = 'positive.review'
    train_neg_set = 'negative.review'
    """
    train_pos_set = 'pos_train.review'
    train_neg_set = 'neg_train.review'
    test_pos_set = 'pos_test.review'
    test_neg_set = 'neg_test.review'
    epochs = 10
    learning_rate = 0.1

    train_set = [train_pos_set, train_neg_set]
    test_set =  [test_pos_set, test_neg_set]
    main(train_set, test_set, epochs, learning_rate)
