import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


class Framework:
    def __init__(self, models=None):
        """
        Initializes model framework
        args:
            models (dict): Sklearn models
                            models = {
                                'model_name':model1(),
                                'model_name':model2(),
                            }
        """
        self.models = models
        self.fitted_models = {}

    def fit(self, X, y, save_dir="./pretrained"):
        """
        Trains and registers the the models in < save_dir >

        args:
            X (2darray): Features data to be used for training
            y (list): Target labels
            save_dir (String): Output directory for pretrained models
        outputs:
            self ``self.fitted_models contains the pretrained models``
        """
        print("\n", "-" * 4, "Fit", "-" * 4)

        for key in self.models.keys():
            self.fitted_models[key] = self.models[key].fit(X, y)
            print(f"{key}: fitted")

        self._save(save_dir=save_dir)

        return self

    @staticmethod
    def predict(X=None, model_path=None):
        """
        Load a fitted model and make a prediction

        args:
            X (2darray): Features data to be used for prediction
            model_path (String): Relative path of the model to be used
        outputs:
            res (list): Predicted labels
        """
        res = None

        with open(path, "rb") as f:
            res = pickle.load(f).predict(X)

        return res

    def _save(self, save_dir):
        """
        Save pretrained models in < save_dir >

        args:
            save_dir (String): output directory for pretrained models

        """
        print("\n", "-" * 4, "Save", "-" * 4)

        for name, model in self.fitted_models.items():
            with open(save_dir + "/" + name + ".pkl", "wb") as f:
                pickle.dump(model, f)

            print(f"Model '{name}.pkl' saved in '{save_dir}'")

    def get_scores(self, X, y, models=None):
        """
        Computes AUC metric using ROC CURVE

        args:
            X (2darray): Features data to be used for prediction
            y (list): Target labels
            models (list): optional list of pretrained model's path.
                           ``If 'None' use the current fitted models, else
                           use the provided models``
        outputs:
            scores (ndarray): Contains AUC scores for each model
        """
        print("\n", "-" * 4, "Score", "-" * 4)

        scores = [[], []]
        roc = {}

        # Compute AUC score an plot ROC Curve
        if models is not None:

            for name in models:
                with open(name, "rb") as f:
                    yhat = pickle.load(f).predict(X)

                    # computes the auc score
                    scores[0].append(name)
                    scores[1].append(roc_auc_score(y, yhat))

                    # computes the roc curves
                    fpr, tpr, _ = roc_curve(y, yhat)
                    roc[name.split("/")[-1]] = (
                        fpr,
                        tpr,
                        round(roc_auc_score(y, yhat), 3),
                    )
        else:
            for name, model in self.fitted_models.items():
                yhat = model.predict(X)

                scores[0].append(name)
                scores[1].append(roc_auc_score(y, yhat))

                fpr, tpr, _ = roc_curve(y, yhat)
                roc[name] = (fpr, tpr, round(roc_auc_score(y, yhat), 3))

        # Plot Roc Curve
        self._plot_roc(roc)

        return scores

    def _plot_roc(self, roc):
        """
        Plots ROC Curves
        
        args:
            roc (tuple) : roc's informations

        """

        for key, value in roc.items():
            plt.plot(value[0], value[1], label=f"{key} - AUC({value[2]})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.axis([-0.01, 1, 0, 1])
        plt.title("ROC Curve")
        plt.xlabel("Specificity")
        plt.ylabel("Sensitivity")

        plt.legend()
        plt.show()
