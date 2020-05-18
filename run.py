import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from src.model import Framework
from src.preprocess import Preprocessing

if __name__ == "__main__":
    """
    Main file to train and/or predict
    """

    # Parameters
    RANDOM_STATE = 42
    METHOD = "undersampling"
    FILE = "./data/creditcard.csv"
    SAVED_DIR = "./pretrained"

    # 1) Training

    # Load & process data
    df = pd.read_csv(FILE)
    Processing = Preprocessing().fit(df, columns_name=["Time", "Amount"])

    # Apply sampling
    Xtrain, Xtest, ytrain, ytest = Processing.get_sample(
        method=METHOD, t_size=0.3, random_state=RANDOM_STATE
    )

    # Train & save models
    my_models = {
        "Logistic_regression": LogisticRegression(random_state=RANDOM_STATE),
        "Naive_Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(),
    }
    learners = Framework(models=my_models).fit(Xtrain, ytrain, SAVED_DIR)

    # Get performance score
    scores = learners.get_scores(Xtest, ytest)
    print(pd.DataFrame({"model_name": scores[0], "AUC": scores[1]}))

    # 2) Inference

    # Predict
    # yhat = Framework.predict(Xtest, "./pretrained/Logistic_regression.pkl")
