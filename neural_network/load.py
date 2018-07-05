#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 訓練データとテストデータを用意
train_data = MNIST('~/tmp/mnist', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_data,
                         batch_size=4,
                         shuffle=True)
test_data = MNIST('~/tmp/mnist', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data,
                         batch_size=4,
                         shuffle=False)

if __name__ == '__main__':
    main()
