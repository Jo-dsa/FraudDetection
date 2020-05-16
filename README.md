# Credit Card Fraud detection
In this project, we implement supervised learning techniques to detect fraud in the creditcard dataset.<br/>
Over-sampling and under-sampling techniques are used.

## Implementation

Language version : Python 3.7.6 <br />
Operating System : MacOS Catalina 10.15.2

## Folder structure

    .
    ├──  data                   # data folder
    ├──  saved_models           # pretrained models folder
    ├──  utils                   
    │   ├── Model.py            # Framework to train models
    │   ├── Preprocess.py       # Processing script to format data
    ├──  requirements.txt       # Dependencies list
    ├──  run.py                 # Main script
    └──  README.md              # Project information


## Getting Started
Install dependencies <br />
```sh
pip install -r requirements.txt
```

Train <br />
```sh
python3 run.py
```
## Current scores
Under-sampling <br />

| model_name          | AUC      |
|---------------------|----------|
| Logistic_regression | 0.931963 |
| Naive_Bayes         | 0.914658 |
| XGBoost             | 0.928630 |

Over-sampling <br />

| model_name          | AUC      |
|---------------------|----------|
| Logistic_regression | 0.948677 |
| Naive_Bayes         | 0.914143 |
| XGBoost             | 0.978393 |
