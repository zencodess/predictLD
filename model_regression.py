#!/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LassoLars

RANDOM_STATE = 69
TOL = 1e-7


def collect_args():
	"""Parse CLI args"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', '-d', type=str, required=True,
						help='dataset path')
	parser.add_argument('--train', '-t', type=bool, default=True,
						help="train on data, and show results")
	parser.add_argument('--split', '-s', type=float, default=0.33,
						help="split data for training and test")
	parser.add_argument('--test', type=bool, default=False,
						help="test on dataset")
	args = parser.parse_args()
	return args


class Model:
	"""Model to train linear classifiers"""

	def __init__(self, data_dir: str, n_labels: int=3, *args, **kwargs):
		self.n_labels = n_labels
		self.filename = data_dir
		self.is_trained = False
		self.is_tested = True
		df = pd.read_csv(self.filename)
		df = df.sample(frac=1).reset_index(drop=True)
		print(f"Loaded dataset : {df.shape}")
		self.dataset = {
			'X': df[df.columns[:-n_labels]],
			'y': df[df.columns[-n_labels:]]
		}
		self.classifiers = [
			LassoLars(*args, **kwargs) for _ in range(n_labels)
		]

	def __str__(self):
		return f"Linear Model: <{self.filename}> dataset"

	def train(self, test_split, *args, **kwargs):
		X, y = self.dataset['X'], self.dataset['y']
		X_train, X_test, y_train, y_test = train_test_split(X, y,
															test_size=test_split,
															random_state=RANDOM_STATE)
		X_train.reset_index(inplace=True, drop=True)
		X_test.reset_index(inplace=True, drop=True)
		y_train.reset_index(inplace=True, drop=True)
		y_test.reset_index(inplace=True, drop=True)
		n_labels = self.n_labels

		print(f"Training data {X_train.shape}, Test data {X_test.shape}")
		[_.fit(X_train, y_train.iloc[:,-n_labels + idx])
		 for idx, _ in enumerate(self.classifiers)]
		self.is_trained = True
		if test_split > 0:
			self.is_tested = self.test(X_test=X_test, y_test=y_test)

	def test(self, *args,
			X_test: pd.DataFrame=None, y_test: pd.DataFrame=None, **kwargs):
		if not self.is_trained:
			print("Error! Train the model first")
			return False

		n_labels = self.n_labels
		if X_test is None or y_test is None:
			print("Loading test dataset")
			X_test, y_test = self.dataset['X'], self.dataset['y']

		results = [_.predict(X_test)
				   for idx, _ in enumerate(self.classifiers)]
		y_hat = np.array(results).T
		self.results = pd.DataFrame(data=y_hat, columns=y_test.columns)
		errors = (np.abs(y_test - self.results) < TOL).astype('int')
		e_data = np.count_nonzero(errors.values, axis=0) / errors.shape[0]
		print(f"Accuracy: {e_data * 100}")

		return True


def main():
	args = collect_args()
	model = Model(args.data)
	if 'train' in args:
		model.train(args.split)
	else:
		model.test()


if __name__ == "__main__":
	main()




	
