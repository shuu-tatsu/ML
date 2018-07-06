#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
import load
import numpy as np
import datetime


class Linear():

    def __init__(self, input_size, target_size, batch_size):
        self.w = np.random.rand(target_size, input_size)
        self.x = np.random.rand(input_size, batch_size)
        self.b = np.random.rand(target_size, batch_size)
        #l1
        #self.w = np.zeros((100, 784))
        #self.x = np.zeros((784, 4))
        #self.b = np.zeros((100, 4))
        #l2
        #self.w = np.zeros((10, 100))
        #self.x = np.zeros((100, 4))
        #self.b = np.zeros((10, 4))

    def linear(self, x):
        self.x = x
        return np.dot(self.w, self.x) + self.b

    def get_layer_parameters(self):
        return self.w, self.x, self.b


class NeuralNetwork(object):

    def __init__(self, batch_size, input_dim_size, hidden_dim_size, output_dim_size):
        self.batch_size = batch_size
        self.input_dim_size = input_dim_size
        self.hidden_dim_size = input_dim_size
        self.output_dim_size = output_dim_size
        #input_dim_size = 784, hidden_dim_size = 100
        self.l1 = Linear(input_dim_size, hidden_dim_size, batch_size) # 入力層から隠れ層へ
        #hidden_dim_size = 100, output_dim_size = 10
        self.l2 = Linear(hidden_dim_size, output_dim_size, batch_size) # 隠れ層から出力層へ

    def forward(self, x):
        x = x.T
        h1 = sigmoid(self.l1.linear(x))
        y = softmax(self.l2.linear(h1))
        return y

    def parameters(self):
        return self.parameters


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x)
    y = exp_x / np.sum(np.exp(x), axis=0, keepdims=True)
    return y

def get_batches(train_features, train_labels, batch_size, shuffle):
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

def train(FILE_TRAIN, epochs, batch_size, input_dim_size, hidden_dim_size, output_dim_size):
    model = NeuralNetwork(batch_size, input_dim_size, hidden_dim_size, output_dim_size)
    # コスト関数と最適化手法を定義
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_loader = load.DataLoader(FILE_TRAIN)
    train_features, train_labels = train_loader.load()
    for epoch in range(epochs):
        for minibatch_features, minibatch_labels in get_batches(train_features,
                                                                train_labels,
                                                                batch_size,
                                                                shuffle=True):
            print('{} EPOCH {} - labels {}'.format(datetime.datetime.today(), epoch, minibatch_labels))
            # 勾配情報をリセット
            #optimizer.zero_grad()
            # 順伝播
            minibatch_predicted_labels = model.forward(minibatch_features)
            print('minibatch_predicted_labels:{}'.format(minibatch_predicted_labels))
            # コスト関数を使ってロスを計算する
            #loss = criterion(minibatch_predicted_labels, minibatch_labels)
            # 逆伝播
            #loss.backward()
            # パラメータの更新
            #optimizer.step()
    print('Finished Training')

def infer(FILE_TEST):
    test_loader = load.DataLoader(FILE_TEST)
    test_features, test_labels = test_loader.load()
    correct = 0
    total = 0
    """
    for data in test_loader:
        inputs, labels = data
        outputs = net(Variable(inputs))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy %d / %d = %f' % (correct, total, correct / total))
    """

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
    train(FILE_TRAIN, EPOCHS, BATCH_SIZE, INPUT_DIM_SIZE, HIDDEN_DIM_SIZE, OUTPUT_DIM_SIZE)
    infer(FILE_TEST)

if __name__ == '__main__':
    main()
