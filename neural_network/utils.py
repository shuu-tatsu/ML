#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
import numpy as np


class SGD(object):

    def __init__(self, model, learning_rate):
        self.l1_w = model.l1_w
        self.l1_b = model.l1_b
        self.l2_w = model.l2_w
        self.l2_b = model.l2_b
        self.learning_rate = learning_rate

    def update(self, grads):
        grad_w1, grad_b1, grad_w2, grad_b2 = grads
        self.l2_w -= self.learning_rate * grad_w2
        self.l2_b -= self.learning_rate * grad_b2
        self.l1_w -= self.learning_rate * grad_w1
        self.l1_b -= self.learning_rate * grad_b1


class CrossEntropyLoss(object):

    def __init__(self, output_dim_size):
        self.output_dim_size = output_dim_size

    def calculate_loss(self,
                       minibatch_predicted_labels,
                       minibatch_labels):
        onehot_labels = onehot_vectorizer(minibatch_labels, self.output_dim_size)
        loss = (-1) * np.dot(onehot_labels, np.log(minibatch_predicted_labels))
        loss = np.sum(loss) / self.output_dim_size
        return loss


def onehot_vectorizer(x, output_dim_size):
    onehot_vec = np.identity(output_dim_size)[x]
    return onehot_vec


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x)
    y = exp_x / np.sum(np.exp(x), axis=0, keepdims=True)
    return y


def get_batches(train_features,
                train_labels,
                batch_size,
                shuffle):
    xs = train_features
    ys = train_labels
    num_samples = len(ys)
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    offset = 0
    while offset < num_samples:
        x = np.take(xs, indices[offset:offset + batch_size], axis=0)
        y = np.take(ys, indices[offset:offset + batch_size], axis=0)
        offset += batch_size
        yield x, y
