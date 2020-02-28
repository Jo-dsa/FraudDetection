import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class Preprocessing(object):
	"""docstring for Preprocess"""
	def __init__(self):
		self.data = None
		self.random_state = None

	def fit(self, X, columns_name=None):
		"""
		Apply transformations on a DataFrame

		args:
			X: pandas DataFrame
			columns_name: list of columns names to apply normalization

		outputs:
			self
		"""

		self.data = X
		self._normalize(columns_name)

		return self

	def _normalize(self, columns_name):
		"""
		Apply a Z normalization on selected columns
		z = (x - mu)/sigma

		args:
			columns_name: list of column names to apply mormalization

		"""

		if not all(e in self.data.columns.values for e in columns_name):
			print(f"Error: One of '{columns_name}' not in DataFrame columns")
		else:
			for e in columns_name:
				sc = StandardScaler()
				self.data[e] = sc.fit_transform(self.data[e].values.reshape(-1, 1)).reshape(-1)
		
	def _oversampling(self):
		"""
		Apply OverSampling using SMOTE on self.data

		outputs:
			x: resampled data
			y: resampled target
		"""
		x_tmp = self.data.drop('Class', axis=1).values
		y_tmp = self.data.Class.values
		
		x, y = SMOTE(sampling_strategy='minority', random_state=self.random_state).fit_sample(x_tmp, y_tmp)
	
		return x, y


	def _undersampling(self):
		"""
		Apply undersampling using RandomUnderSampler on self.data

		outputs:
			x: resampled data
			y: resampled target
		"""
		x_tmp = self.data.drop('Class', axis=1).values
		y_tmp = self.data.Class.values

		x, y = RandomUnderSampler(random_state=self.random_state).fit_sample(x_tmp, y_tmp)
		
		return x, y

	def _train_test(self, x, y, t_size):
		"""
		Split arrays or matrices into random train and test subsets

		args:
			x: matrix to split
			y: target list to split
			t_size: proportion of test sample (float) between 0 and 1
		outputs:
			tuple containing train-test split of inputs.
		"""
		
		return train_test_split(x, y, test_size=t_size, random_state=self.random_state)

	def get_sample(self, method='overampling', t_size=.3, random_state=42 ):
		"""
		This function return train and test subsets given the sampling method

		args:
			method: sampling method 'oversampling' or 'undersampling'
			t_size: proportion of test sample (float) between 0 and 1
			random_state: random state number, helps to get a consistent experimentation

		return:
			Xtrain: train sample (matrix)			
			Xtest: test sample (matrix)
			ytrain: test sample (list)
			ytest: target sample (list)
		"""
		self.random_state = random_state
		x, y = None, None
		Xtrain, Xtest, ytrain, ytest = None, None, None, None

		if method not in ['oversampling','undersampling']:
			print(f"Error: Unkown method '{method}', available are ['oversampling', 'undersampling']")
		else:	
			if method == 'oversampling':
				print('-'*4,'OVER-SAMPLING','-'*4)
				x, y = self._oversampling()
			else:
				print('-'*4,'UNDER-SAMPLING','-'*4)
				x, y = self._undersampling()

			Xtrain, Xtest, ytrain, ytest = self._train_test(x, y, t_size)

		return Xtrain, Xtest, ytrain, ytest

