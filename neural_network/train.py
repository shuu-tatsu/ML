#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
import load
import inference_test
import utils
import numpy as np
import datetime


class Linear(object):

    def __init__(self,
                 input_size,
                 target_size,
                 batch_size):
        self.w = np.random.rand(target_size, input_size)
        self.b = np.random.rand(target_size, 1)
        self.target_size = target_size
        self.batch_size = batch_size

    def linear(self, x):
        ones = np.ones((x.shape[1], 1))
        wx = np.dot(self.w, x).reshape(self.target_size, -1)
        b = np.dot(self.b, ones.T)
        return wx + b

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
        z1 = utils.sigmoid(self.l1.linear(x))
        y = utils.softmax(self.l2.linear(z1))
        return z1, y

    def backward(self, x, z1, y, d):
        ones = np.ones((self.batch_size, 1))

        delta2 = y - d
        grad_w2 = np.dot(delta2, z1.T) / self.batch_size
        grad_b2 = np.dot(delta2, ones) / self.batch_size

        sigmoid_dash = z1 * (1 - z1)

        delta1 = sigmoid_dash * np.dot(self.l2_w.T, delta2)
        grad_w1 = np.dot(delta1, x) / self.batch_size
        grad_b1 = np.dot(delta1, ones) / self.batch_size

        grads = [grad_w1, grad_b1, grad_w2, grad_b2]
        return grads


def train(file_train,
          file_test,
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
    cross_entropy = utils.CrossEntropyLoss(output_dim_size)
    optimizer = utils.SGD(model, learning_rate)
    train_loader = load.DataLoader(file_train)
    train_features, train_labels = train_loader.load()
    for epoch in range(epochs):
        for minibatch_features, minibatch_labels in utils.get_batches(train_features,
                                                                      train_labels,
                                                                      batch_size,
                                                                      shuffle=True):
            # 順伝播
            minibatch_features_reshaped = minibatch_features.T
            z1, minibatch_predicted_labels = model.forward(minibatch_features_reshaped)
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
        inference_test.infer(file_test=file_test,
                             model_trained=model)

    print('Finished Training')
    return model


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

    model_trained = train(file_train=FILE_TRAIN,
                          file_test=FILE_TEST,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          input_dim_size=INPUT_DIM_SIZE,
                          hidden_dim_size=HIDDEN_DIM_SIZE,
                          output_dim_size=OUTPUT_DIM_SIZE,
                          learning_rate=LEARNING_RATE)

    #inference_test.infer(file_test=FILE_TEST,
    #                     model_trained=model_trained)


if __name__ == '__main__':
    main()
