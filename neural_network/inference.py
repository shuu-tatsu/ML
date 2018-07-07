#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
import load
import numpy as np


def infer(file_test):
    test_loader = load.DataLoader(file_test)
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


if __name__ == '__main__':
    main()
