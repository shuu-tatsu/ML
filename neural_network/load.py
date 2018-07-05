#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class DataLoader():
    def __init__(self, data_path, batch_size, shuffle):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle

    def load(self):
        with open(self.data_path, 'r') as r:
            lines = r.readlines()
            data_features, data_labels = read_data(lines)
        return data_features, data_labels

def read_data(lines):
    features = [line[:-1].split(',')[1:] for line in lines]
    labels = [line.split(',')[0] for line in lines]
    return features, labels

def main(train_data_path, test_data_path):
    train_loader = DataLoader(train_data_path, batch_size=4, shuffle=True)
    train_features, train_labels = train_loader.load()

    test_loader = DataLoader(test_data_path, batch_size=4, shuffle=False)
    test_features, test_labels = test_loader.load()

    print(test_features)
    print(test_labels)
    

if __name__ == '__main__':
    #train_data_path = './mnist/MNIST-csv/train.csv'
    #test_data_path = './mnist/MNIST-csv/test.csv'
    train_data_path = './mnist/MNIST-csv/toy_train.csv'
    test_data_path = './mnist/MNIST-csv/toy_test.csv'
    main(train_data_path, test_data_path)
