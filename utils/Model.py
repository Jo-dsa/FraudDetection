import pickle
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class Framework():

	def __init__(self, models=None):
		"""
		Initialize the model framework
		args:
			models: Dictionary of callable machine learning models
					ex: dico = {
							'model_name':model_1(),
							'model_name':model_2(),
							...
						}
		"""
		self.models = models
		self.fitted_models = {}

	def fit(self, X, y, saved_dir='./saved_dir'):
		"""
		Train all models in self.models and save them in './saved_models'

		args:
			X: features data (matrix)
			y: target data (list)
			saved_dir: directory where models should be saved
		outputs:
			return self instance. Fitted models are in 'self.fitted_models'
		"""
		print('\n','-'*4,'Fit','-'*4)

		for key in self.models.keys():
			self.fitted_models[key] = self.models[key].fit(X, y)
			print(f"{key}: fitted")

		self._save(dir=saved_dir)

		return self

	@staticmethod
	def predict(X=None, path=None):
		"""
		Load a fitted model and make a prediction

		args:
			X: features matrix to predict
			path: path to the desire file model (string)
		outputs:
			list of prediction
		"""

		with open(path, 'rb') as f:
			return pickle.load(f).predict(X)
		

	def _save(self, dir):
		"""
		Save all the models in 'dir' directory

		args:
			dir: directory to save models (string)

		"""
		print('\n','-'*4,'Save','-'*4)

		for name, model in self.fitted_models.items():
			with open(dir+'/'+name+'.pkl', 'wb') as f:
				pickle.dump(model, f)

			print(f"Model '{name}.pkl' saved in '{dir}'")
		

	def scores(self, X=None, y=None, models=None):
		"""
		Computes AUC metric using ROC CURVE

		args:
			X: features matrix to predict
			y: real labels (array)
			models: path to pretrained models (array).
					If not None use pretrained models for path 
					instead of fitted model in self Object
		outputs:
			table: AUC score for each model (ndarray)
		"""
		print('\n','-'*4,'Score','-'*4)

		table = [[],[]]
		roc_tmp = {}

		

		#Compute AUC score an plot ROC Curve
		if models is not None:

			for name in models:				
				with open(name, 'rb') as f:
					yhat = pickle.load(f).predict(X)

					table[0].append(name)
					table[1].append(roc_auc_score(y, yhat))

					fpr, tpr, _ = roc_curve(y, yhat)
					roc_tmp[name.split('/')[-1]] = (fpr, tpr, round(roc_auc_score(y, yhat), 3))
		else:
			for name, model in self.fitted_models.items():
				yhat = model.predict(X)

				table[0].append(name)
				table[1].append(roc_auc_score(y, yhat))

				fpr, tpr, _ = roc_curve(y, yhat)
				roc_tmp[name] = (fpr, tpr, round(roc_auc_score(y, yhat),3))

		#Plot Roc Curve
		self._plot_roc(roc_tmp)

		return table

	def _plot_roc(self, roc_tmp):
		"""
		Internal function use to plot ROC Curve
		
		args:
			roc_tmp: tuple of roc informations

		"""

		for key, value in roc_tmp.items():
			plt.plot(value[0], value[1], label=f"{key} - AUC({value[2]})")

		plt.plot([0, 1], [0, 1], 'k--')			
		plt.axis([-0.01, 1, 0, 1])
		plt.title("ROC Curve")
		plt.xlabel("Specificity")
		plt.ylabel("Sensitivity")

		plt.legend()
		plt.show()

