#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy.random import *
import numpy as np
import math
import logging
import random

class GaussianInitializer(object):
	def __init__(self):
		self.dim = 0

	def apply(self, w):
		self.dim = w.shape[0]
		w = randn(self.dim)


def sigmoid(x):
	sigmoid = math.exp(x) / (1 + math.exp(x))
	return sigmoid


class LogisticRegression(object):

	def __init__(self, w):
		self.w = w

	def grad_mini_batch(self, x_mini_batch, y_mini_batch):
		target = np.dot(x_mini_batch, (sigmoid(np.dot((self.w).T, x_mini_batch)) - y_mini_batch))
		return target

	def forward(self, x_mini_batch):
		ys_predict = sigmoid(np.dot((self.w).T, x_mini_batch))
		a = np.dot((self.w).T, x_mini_batch)
		return ys_predict


class SGD(object):

	def __init__(self, learning_rate):
		self.learning_rate = learning_rate

	def cross_entropy_error_func(self, model, x_mini_batch, y_mini_batch):
		model.w -= self.learning_rate * model.grad_mini_batch(x_mini_batch, y_mini_batch)


class Evaluator(object):
	"""
	def __init__(self, xs_test, ys_test, model):
		self.xs_test = xs_test
		self.ys_test = ys_test
		self.model = model

	def evaluate(self):
		loss = 0
		correct = 0
		for i in range(len(self.ys_test)):
			ys_predict = self.model.forward(self.xs_test[i])
			loss += culc_loss(ys_predict, self.ys_test[i])
			correct += count_correct(ys_predict, self.ys_test[i])
		accuracy = correct / len(self.ys_test)
		return loss, accuracy
	"""
	def __init__(self, model):
		self.model = model

	def evaluate(self,xs_test,ys_test):
		ys_predict = self.model.forward(xs_test)
		loss = culc_loss(ys_predict, ys_test)
		accuracy = count_correct(ys_predict, ys_test)
		return loss, accuracy

def count_correct(y_predict_mini_batch, y_mini_batch):
	if y_predict_mini_batch > 0.5:
		y_predict_label = 1
	else:
		y_predict_label = 0
	if y_predict_label == y_mini_batch:
		correct = 1
	else:
		correct = 0
	return correct


def culc_loss(y_predict_mini_batch, y_mini_batch):
	loss = abs(y_predict_mini_batch - y_mini_batch)
	return loss


def get_mini_batches(xs_train, ys_train):
	for i in range(len(ys_train)):
		yield xs_train[i], ys_train[i]


def train(xs_train, ys_train, xs_test, ys_test, epochs, learning_rate):
	xs_train = np.load(xs_train)
	ys_train = np.load(ys_train)
	xs_test = np.load(xs_test)
	ys_test = np.load(ys_test)
	dim = xs_train[0].shape[0]
	w = np.empty((dim,), dtype=np.float16)
	initializer = GaussianInitializer()
	initializer.apply(w)
	model = LogisticRegression(w)
	optimizer = SGD(learning_rate)

	def process(xs_train, ys_train, xs_test, ys_test):
		evaluator = Evaluator(model)
		for x_mini_batch, y_mini_batch in get_mini_batches(xs_train, ys_train):
			loss, accuracy = evaluator.evaluate(x_mini_batch,y_mini_batch)
			optimizer.cross_entropy_error_func(model, x_mini_batch, y_mini_batch)
		return loss, accuracy

	for epoch in range(1, epochs + 1):
		loss, accuracy = process(xs_train, ys_train, xs_test, ys_test)
		logging.info(
			"[{}] epoch {} - #samples: {}, loss: {:.8f}, accuracy: {:.8f}"
			.format("train", epoch, len(ys_train), loss, accuracy))


def main(xs_train, ys_train, xs_test, ys_test, epochs, learning_rate):
	train(xs_train, ys_train, xs_test, ys_test, epochs, learning_rate)


if __name__ == '__main__':
	logging.basicConfig(
	format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
	epochs = 30
	learning_rate = 0.05
	xs_train = 'xs_divide_train.npy'
	ys_train = 'ys_divide_train.npy'
	xs_test = 'xs_divide_test.npy'
	ys_test = 'ys_divide_test.npy'
	main(xs_train, ys_train, xs_test, ys_test, epochs, learning_rate)
