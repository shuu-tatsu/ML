#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
import load
import numpy as np
import torch


def infer(file_test, model_trained):
    test_loader = load.DataLoader(file_test)
    test_features, test_labels = test_loader.load()
    correct = 0
    total = test_labels.size()
    for feature, label in test_features, test_labels:
        _, predicted_tensor = model_trained.forward(feature)
        _, predicted_label = torch.max(predicted_label, 0)
        if predicted_label == label:
            correct += 1
    print('Accuracy %d / %d = %f' % (correct, total, correct / total))


if __name__ == '__main__':
    main()
