#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class DataLoader():

    def __init__(self, data_path):
        self.data_path = data_path

    def load(self):
        with open(self.data_path, 'r') as r:
            lines = r.readlines()
            data_features, data_labels = read_data(lines)
        return data_features, data_labels


def read_data(lines):
    features = [line[:-1].split(',')[1:] for line in lines]
    labels = [line.split(',')[0] for line in lines]
    return features, labels

def main():
    #FILE_TRAIN = './mnist/MNIST-csv/train.csv'
    #FILE_TEST = './mnist/MNIST-csv/test.csv'
    FILE_TRAIN = './mnist/MNIST-csv/toy_train.csv'
    FILE_TEST = './mnist/MNIST-csv/toy_test.csv'

    train_loader = DataLoader(FILE_TRAIN)
    train_features, train_labels = train_loader.load()

    test_loader = DataLoader(FILE_TEST)
    test_features, test_labels = test_loader.load()

    print(train_features, train_labels)

if __name__ == '__main__':
    main()
