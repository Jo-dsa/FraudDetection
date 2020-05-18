import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class Preprocessing(object):
    """
	Implements processing step for data
	"""

    def __init__(self):
        self.data = None
        self.random_state = None

    def fit(self, X, columns_name=None):
        """
		Transforms the provided data

		args:
			X (DataFrame): Features data
			columns_name (list): List of column names on which to apply normalization
		outputs:
			self
		"""

        self.data = X
        self._normalize(columns_name)

        return self

    def _normalize(self, columns_name):
        """
		Applies a Z normalization on selected columns
		z = (x - mu)/sigma

		args:
			columns_name (list): List of column names on which to apply normalization

		"""
        if not all(name in self.data.columns.values for name in columns_name):
            print(f"Error: One of '{columns_name}' not in DataFrame columns")
        else:
            for name in columns_name:
                sc = StandardScaler()
                self.data[name] = sc.fit_transform(
                    self.data[name].values.reshape(-1, 1)
                ).reshape(-1)

    def _oversampling(self):
        """
		Applies OverSampling using SMOTE on self.data

		outputs:
			x (2darray): resampled features
			y (1darray): resampled targets
		"""
        x_tmp = self.data.drop("Class", axis=1).values
        y_tmp = self.data.Class.values

        x, y = SMOTE(
            sampling_strategy="minority", random_state=self.random_state
        ).fit_sample(x_tmp, y_tmp)

        return x, y

    def _undersampling(self):
        """
		Applies undersampling using RandomUnderSampler on self.data

		outputs:
			x (2darray): resampled features
			y (1darray): resampled targets
		"""
        x_tmp = self.data.drop("Class", axis=1).values
        y_tmp = self.data.Class.values

        x, y = RandomUnderSampler(random_state=self.random_state).fit_sample(
            x_tmp, y_tmp
        )

        return x, y

    def _train_test(self, x, y, t_size):
        """
		Split features and targets into random train and test subsets

		args:
			x (2darray): features data to split
			y (1darray): target labels to split
			t_size (float): proportion of test sample. 0 < t_size < 1
		outputs:
			tuple that contains train/test split
		"""

        return train_test_split(x, y, test_size=t_size, random_state=self.random_state)

    def get_sample(self, method="overampling", t_size=0.3, random_state=42):
        """
		Returns train and test subsets given the sampling method

		args:
			method (String): Sampling method. 'oversampling' || 'undersampling'
			t_size (Float): Proportion of test sample. 0 < t_size < 1
			random_state (int): random state number. ``Fix random behavior``

		return:
			Xtrain (2darray): features sample for train
			Xtest (1darray): features sample for test
			ytrain (2darray): target sample for train
			ytest (1darray): target sample for test
		"""
        self.random_state = random_state
        x, y = None, None
        Xtrain, Xtest, ytrain, ytest = None, None, None, None

        if method not in ["oversampling", "undersampling"]:
            print(
                f"Error: Unkown method '{method}', available are ['oversampling', 'undersampling']"
            )
        else:
            if method == "oversampling":
                print("-" * 4, "OVER-SAMPLING", "-" * 4)
                x, y = self._oversampling()
            else:
                print("-" * 4, "UNDER-SAMPLING", "-" * 4)
                x, y = self._undersampling()

            Xtrain, Xtest, ytrain, ytest = self._train_test(x, y, t_size)

        return Xtrain, Xtest, ytrain, ytest
