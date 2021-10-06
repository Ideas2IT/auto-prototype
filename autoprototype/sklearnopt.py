"""SKlearn Hyperopt class."""
import optuna
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm
import sklearn.tree
import sklearn.linear_model


class HyperparamOpt:
    """Sklearn hyper parameter optimization class.

    Args:
        x : independent variables
        y : target variable
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def objective(self, trial):
        """Setting the objective function.

        Args:
            trial : optuna trial

        Returns:
            accuracy : model accuracy for optimization
        """
        classifier_name = trial.suggest_categorical(
            "classifier",
            [
                "SVC",
                "DecisionTree",
                "RandomForest",
                "Linear",
                "Ridge",
                "Lasso",
                "Logistic",
            ],
        )
        if classifier_name == "SVC":
            svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
            classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
        elif classifier_name == "Linear":
            normalize = trial.suggest_categorical("normalize",
                                                  ["True", "False"])
            classifier_obj = sklearn.linear_model.LinearRegression(
                normalize=normalize)
        elif classifier_name == "Ridge":
            alpha = trial.suggest_uniform("alpha", 0, 1)
            classifier_obj = sklearn.linear_model.Ridge(alpha=alpha)
        elif classifier_name == "Logistic":
            max_iter = trial.suggest_int("max_iter", 1000, 1500)
            classifier_obj = sklearn.linear_model.LogisticRegression(
                max_iter=max_iter)
        elif classifier_name == "Lasso":
            alpha = trial.suggest_uniform("alpha", 0, 1)
            classifier_obj = sklearn.linear_model.Lasso(alpha=alpha)
        elif classifier_name == "DecisionTree":
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 3)
            classifier_obj = sklearn.tree.DecisionTreeClassifier(
                min_samples_leaf=min_samples_leaf)
        else:
            rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
            n_estimators = trial.suggest_int("n_estimators", 2, 12)
            classifier_obj = sklearn.ensemble.RandomForestClassifier(
                max_depth=rf_max_depth, n_estimators=n_estimators)
        cross_validation_k = trial.suggest_int("cross_validation_k", 2, 10)
        score = sklearn.model_selection.cross_val_score(classifier_obj,
                                                        self.x,
                                                        self.y,
                                                        n_jobs=-1,
                                                        cv=cross_validation_k)
        accuracy = score.mean()
        return accuracy

    def get_best_params(self, n_trials):
        """Gets the best parameter and the best model architecture.

        Args:
             n_trials : number of trials

        Returns:
            best_trial : the best trial.
            best_paramteres : the best parameters got after optimization.
            best_value : the best loss value corresponding to the trial.
        """
        study = optuna.create_study(
            pruner="PercentilePruner",
            direction="maximize",
            study_name="Ankan Study",
        )
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_trial, study.best_params, study.best_value
