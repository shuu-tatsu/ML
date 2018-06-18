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
    return xs_train, ys_train


class GaussianInitializer(object):
    def __init__(self):
        self.dim = 0

    def apply(self, w):
        self.dim = w.shape[0]
        w = randn(self.dim)


def sigmoid(x):
    sigmoid = math.exp(x) / (1 + math.exp(x))
    return sigmoid


class LogisticRegression(object):

    def __init__(self, w):
        self.w = w

    def grad_batch(self, xs_train, ys_train):
        target = [0 for _ in range(200000)]
        for i in range(len(xs_train)):
            target = target + (xs_train[i] * (sigmoid(np.dot((self.w).T, xs_train[i]) - ys_train[i])))
        return target

    def grad_mini_batch(self, x_mini_batch, y_mini_batch):
        target = (x_mini_batch * (sigmoid(np.dot((self.w).T, x_mini_batch) - y_mini_batch)))
        return target

    def forward(self, x_mini_batch):
        ys_predict = (sigmoid(np.dot((self.w).T, x_mini_batch)))
        return ys_predict


class SGD(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def cross_entropy_error_func(self, model, x_mini_batch, y_mini_batch):
        model.w = model.w - self.learning_rate * model.grad_mini_batch(x_mini_batch, y_mini_batch)


def count_correct(y_predict_mini_batch, y_mini_batch):
    if y_predict_mini_batch > 0.5:
        y_predict_label = 1
    else:
        y_predict_label = 0
    if y_predict_label == y_mini_batch:
        correct = 1
    else:
        correct = 0
    return correct


def culc_loss(y_predict_mini_batch, y_mini_batch):
    loss = abs(y_predict_mini_batch - y_mini_batch)
    return loss


def get_mini_batches(xs_train, ys_train):
    for i in range(len(ys_train)):
        yield xs_train[i], ys_train[i]

def train(train_set, epochs, learning_rate):
    xs_train, ys_train = load(train_set)
    dim = xs_train[0].shape[0]
    w = np.empty((dim,), dtype=np.float16)
    initializer = GaussianInitializer()
    initializer.apply(w)
    model = LogisticRegression(w)
    optimizer = SGD(learning_rate)

    def process(xs_train, ys_train):
        loss = 0.0
        correct = 0
        for x_mini_batch, y_mini_batch in get_mini_batches(xs_train, ys_train):
            y_predict_mini_batch = model.forward(x_mini_batch)
            loss += culc_loss(y_predict_mini_batch, y_mini_batch)
            correct += count_correct(y_predict_mini_batch, y_mini_batch)
            optimizer.cross_entropy_error_func(model, x_mini_batch, y_mini_batch)
        accuracy = correct / len(ys_train)
        return loss, accuracy

    for epoch in range(1, epochs + 1):
        loss, accuracy = process(xs_train, ys_train)
        logging.info(
            "[{}] epoch {} - #samples: {}, loss: {:.8f}, accuracy: {:.8f}"
            .format("train", epoch, len(ys_train), loss, accuracy))

def test(test_set):
    test_set = None


def main(train_set, test_set, epochs, learning_rate):
    train(train_set, epochs, learning_rate)
    test(test_set)


if __name__ == '__main__':
    logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    train_pos_set = 'positive.review'
    train_neg_set = 'negative.review'
    """
    train_pos_set = 'pos_train.review'
    train_neg_set = 'neg_train.review'
    """
    test_pos_set = 'pos_test.review'
    test_neg_set = 'neg_test.review'
    epochs = 10
    learning_rate = 0.1

    train_set = [train_pos_set, train_neg_set]
    test_set =  [test_pos_set, test_neg_set]
    main(train_set, test_set, epochs, learning_rate)
