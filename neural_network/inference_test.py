#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
import load
import numpy as np


def infer(file_test, model_trained):
    test_loader = load.DataLoader(file_test)
    test_features, test_labels = test_loader.load()
    correct = 0
    total = len(test_labels)
    for feature, label in zip(test_features, test_labels):
        feature = np.array(feature)
        label = np.array(label)
        predicted_tensor = model_trained.forward_inference(feature)
        print('predicted_tensor:{}'.format(predicted_tensor))
        predicted_label = np.argmax(predicted_tensor)
        print('predicted_label:{}'.format(predicted_label))
        if predicted_label == label:
            correct += 1
    print('Accuracy %d / %d = %f' % (correct, total, correct / total))


if __name__ == '__main__':
    main()
