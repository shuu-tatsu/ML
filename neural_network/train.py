#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
import load
import inference
import numpy as np
import datetime


class Linear(object):

    def __init__(self,
                 input_size,
                 target_size,
                 batch_size):
        self.w = np.random.rand(target_size, input_size)
        self.x = np.random.rand(input_size, batch_size)
        self.b = np.random.rand(target_size, batch_size)

    def linear(self, x):
        self.x = x
        return np.dot(self.w, self.x) + self.b

    def get_layer_parameters(self):
        return self.w, self.b


class NeuralNetwork(object):

    def __init__(self,
                 batch_size,
                 input_dim_size,
                 hidden_dim_size,
                 output_dim_size):
        self.batch_size = batch_size
        self.input_dim_size = input_dim_size
        self.hidden_dim_size = input_dim_size
        self.output_dim_size = output_dim_size
        # 入力層から隠れ層へ
        self.l1 = Linear(input_dim_size,
                         hidden_dim_size,
                         batch_size)
        self.l1_w, self.l1_b = self.l1.get_layer_parameters()
        # 隠れ層から出力層へ
        self.l2 = Linear(hidden_dim_size,
                         output_dim_size,
                         batch_size)
        self.l2_w, self.l2_b = self.l2.get_layer_parameters()

    def forward(self, x):
        x = x.T
        z1 = sigmoid(self.l1.linear(x))
        y = softmax(self.l2.linear(z1))
        return z1, y

    def backward(self, x, z1, y, d):
        delta2 = y - d
        grad_w2 = np.dot(delta2, z1.T) / self.batch_size
        grad_b2 = delta2 / self.batch_size

        sigmoid_dash = z1 * (1 - z1)
        delta1 = sigmoid_dash * np.dot(self.l2_w.T, delta2)
        grad_w1 = np.dot(delta1, x) / self.batch_size
        grad_b1 = delta1 / self.batch_size
        grads = [grad_w1, grad_b1, grad_w2, grad_b2]
        return grads


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


def train(file_train,
          epochs,
          batch_size,
          input_dim_size,
          hidden_dim_size,
          output_dim_size,
          learning_rate):
    model = NeuralNetwork(batch_size,
                          input_dim_size,
                          hidden_dim_size,
                          output_dim_size)
    cross_entropy = CrossEntropyLoss(output_dim_size)
    optimizer = SGD(model, learning_rate)
    train_loader = load.DataLoader(file_train)
    train_features, train_labels = train_loader.load()
    for epoch in range(epochs):
        for minibatch_features, minibatch_labels in get_batches(train_features,
                                                                train_labels,
                                                                batch_size,
                                                                shuffle=True):
            # 順伝播
            z1, minibatch_predicted_labels = model.forward(minibatch_features)
            # 評価用にLOSSを算出
            loss = cross_entropy.calculate_loss(minibatch_predicted_labels, minibatch_labels)
            print('[{}] EPOCH {} - LOSS {:.8f}'.format(datetime.datetime.today(), epoch, loss))
            # 逆伝播
            grads = model.backward(x=minibatch_features,
                                   z1=z1,
                                   y=minibatch_predicted_labels,
                                   d=minibatch_labels)
            # パラメータの更新
            optimizer.update(grads)
    print('Finished Training')


def main():
    #FILE_TRAIN = './mnist/MNIST-csv/train.csv'
    #FILE_TEST = './mnist/MNIST-csv/test.csv'
    FILE_TRAIN = './mnist/MNIST-csv/toy_train.csv'
    FILE_TEST = './mnist/MNIST-csv/toy_test.csv'
    EPOCHS = 5
    BATCH_SIZE = 4
    INPUT_DIM_SIZE = 28 * 28
    HIDDEN_DIM_SIZE = 100
    OUTPUT_DIM_SIZE = 10
    LEARNING_RATE = 0.01

    train(file_train=FILE_TRAIN,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          input_dim_size=INPUT_DIM_SIZE,
          hidden_dim_size=HIDDEN_DIM_SIZE,
          output_dim_size=OUTPUT_DIM_SIZE,
          learning_rate=LEARNING_RATE)

    inference.infer(file_test=FILE_TEST)


if __name__ == '__main__':
    main()
