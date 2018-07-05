#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

def train():
    for epoch in range(3):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # Variableに変換
            inputs, labels = Variable(inputs), Variable(labels)
            # 勾配情報をリセット
            optimizer.zero_grad()
            # 順伝播
            outputs = net(inputs)
            # コスト関数を使ってロスを計算する
            loss = criterion(outputs, labels)
            # 逆伝播
            loss.backward()
            # パラメータの更新
            optimizer.step()
            running_loss += loss.data[0]
            if i % 5000 == 4999:
                print('%d %d loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
    print('Finished Training')

def infer():
    correct = 0
    total = 0
    for data in test_loader:
        inputs, labels = data
        outputs = net(Variable(inputs))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy %d / %d = %f' % (correct, total, correct / total))

def main():
    net = Net()
    # コスト関数と最適化手法を定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    train()
    infer()

if __name__ == '__main__':
    main()
