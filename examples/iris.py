from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from OptunaHPO.sklearnHPO import HyperparamOpt
X, y = load_iris(return_X_y=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
classes = np.unique(y)
trials = 50
hpo = HyperparamOpt(X_train,y_train)
trial , params , accuracy = hpo.get_best_params(n_trials=trials)

print("\n")
print("This is the best trial",trial)
print("\n")
print("This is the best params",params)
print("\n")
print("This is the accuracy",accuracy)
