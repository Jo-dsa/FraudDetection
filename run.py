from utils.Model import Framework
from utils.Preprocess import Preprocessing

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

if __name__ == '__main__':
	"""
		Main file to train and/or predict

	"""
	# Parametres
	RANDOM_STATE = 42
	METHOD = 'undersampling'
	FILE = './data/creditcard.csv'
	SAVED_DIR = './saved_models'

	# Read data
	df = pd.read_csv(FILE)
	# Clean data
	Processing = Preprocessing().fit(df, columns_name=['Time','Amount'])
	# Get sample using oversampling or undersampling
	Xtrain, Xtest, ytrain, ytest =  Processing.get_sample(method=METHOD, t_size=.3, random_state=RANDOM_STATE )


	# Train & save models	
	my_models = {
		'Logistic_regression':LogisticRegression(random_state = RANDOM_STATE),
		'Naive_Bayes':GaussianNB(),
		'XGBoost':XGBClassifier(),
	}
	ensemble_learners = Framework(models=my_models).fit(Xtrain, ytrain, SAVED_DIR)
	
	# Get performance
	s = ensemble_learners.scores(Xtest, ytest)
	print(pd.DataFrame({'model_name':s[0],'AUC':s[1]}))


	# predict
	yhat = Framework.predict(Xtest, "./saved_models/Logistic_regression.pkl")





