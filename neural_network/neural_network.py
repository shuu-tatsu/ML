#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
import load
import numpy as np
import datetime

"""
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 50) # 入力層から隠れ層へ
        self.l2 = nn.Linear(50, 10) # 隠れ層から出力層へ

    def forward(self, x):
        x = x.view(-1, 28 * 28) # テンソルのリサイズ: (N, 1, 28, 28) --> (N, 784)
        x = self.l1(x)
        x = self.l2(x)
        return x

    def parameters(self):
"""

def get_batches(train_features, train_labels, batch_size, shuffle=False):
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

def train(FILE_TRAIN, epochs, batch_size):
    #net = Net()
    # コスト関数と最適化手法を定義
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.01)
    train_loader = load.DataLoader(FILE_TRAIN)
    train_features, train_labels = train_loader.load()
    for epoch in range(epochs):
        for minibatch_features, minibatch_labels in get_batches(train_features, train_labels, batch_size, shuffle=train):
            inputs = minibatch_features
            labels = minibatch_labels
            print('{} EPOCH {} - labels {}'.format(datetime.datetime.today(), epoch, labels))
            # 勾配情報をリセット
            #optimizer.zero_grad()
            # 順伝播
            #outputs = net(inputs)
            # コスト関数を使ってロスを計算する
            #loss = criterion(outputs, labels)
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
    train(FILE_TRAIN, EPOCHS, BATCH_SIZE)
    infer(FILE_TEST)

if __name__ == '__main__':
    main()
