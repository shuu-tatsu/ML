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
        feature_reshaped = feature.reshape(model_trained.input_dim_size, -1)
        _, predicted_tensor = model_trained.forward(feature_reshaped)
        predicted_label = np.argmax(predicted_tensor)
        #print('Predicted_label:{} Correct_label:{}'.format(predicted_label, label))
        if predicted_label == label:
            correct += 1
    accuracy = correct / total
    return accuracy
