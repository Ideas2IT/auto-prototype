"""tf keras optimization class"""
import urllib
import datetime
import warnings
from typing import Optional
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
import tensorflow as tf

# TODO(crcrpar): Remove the below three lines once everything is ok.
# Register a global custom opener to avoid HTTP Error 403: Forbidden when downloading MNIST.
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


class KerasHPO():
    """Class for tf keras hyper-parameter optimization."""
    def __init__(self,
                 train,
                 target,
                 classes,
                 EPOCHS: Optional[int] = 5,
                 steps_per_epoch: Optional[int] = 216,
                 loss: Optional[str] = "sparse_categorical_crossentropy",
                 max_nconv: Optional[int] = 5,
                 max_conv_filters: Optional[int] = 256,
                 arch: Optional[str] = None,
                 max_fully_conn_layers: Optional[int] = 2,
                 n_ann_layers: Optional[int] = 5,
                 max_units_fcl: Optional[int] = 1000,
                 input_shape: Optional[tuple] = None):
        """Initializing the arguments.

        Args :
            trian : independent variable
            target : labels or class variables
            EPOCHS : number of epochs in each trial
            steps_per_epoch : number of steps per epoch
            loss : the loss function
            max_nconv : max number of conv layers
            max_nconv_filters : max filter size in each conv layer
            arch : choice of architecture
            max_fully_conn_layers : number of fully connected layer in conv
            n_ann_layers : max layers in ann model
            max_units_fcl : maximum number of units in the fully connected layers cnn
            input_shapre : shape of the input data.
        """
        self.EPOCHS = EPOCHS
        self.ds_train = train
        self.steps_per_epoch = steps_per_epoch
        self.loss = loss
        self.max_fully_conn_layers = max_fully_conn_layers
        self.n_ann_layers = n_ann_layers
        self.max_nconv = max_nconv
        self.max_conv_filters = max_conv_filters
        self.target = target
        self.max_units_fcl = max_units_fcl
        self.classes = classes
        self.arch = arch
        self.input_shape = input_shape

    def create_model(self, trial):
        """Creates the ANN model structure.

        Args:
            trial : optuna trial

        Returns:
            model : The trial model for each iterations in the trial.
        """

        # Hyperparameters to be tuned by Optuna.
        n_layers = trial.suggest_int("n_layers", 1, self.n_ann_layers)
        model = tf.keras.Sequential()
        for i in range(n_layers):
            units = trial.suggest_int("n_units_l{}".format(i),
                                      4,
                                      128,
                                      log=True)
            model.add(tf.keras.layers.Dense(units=units))
            dropout = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
            model.add(tf.keras.layers.Dropout(rate=dropout))
        model.add(tf.keras.layers.Dense(self.classes, activation="softmax"))

        # We compile our model with a sampled learning rate.
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer_name = trial.suggest_categorical("optimizer", [Adam, RMSprop, SGD])
        model.compile(
            loss=self.loss,
            optimizer=optimizer_name(learnig_rate=lr),
            metrics=["accuracy"],
        )

        return model

    def cnn_model(self, trial):
        """Creates the CNN model structure.

        Args:
            trial : optuna trial

        Returns:
            model : The trial cnn model for each iterations in the trial.
        """

        # Hyperparameters to be tuned by Optuna.
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

        nconv = trial.suggest_int("nconv", 2, self.max_nconv)
        print(self.input_shape)
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=trial.suggest_categorical("kernel_size", [3, 5]),
                strides=trial.suggest_categorical("strides", [1, 2]),
                padding="same",
                activation=trial.suggest_categorical(
                    "activation", ["relu", "linear", "tanh"]),
                input_shape=self.input_shape,
            ))
        model.add(tf.keras.layers.BatchNormalization(axis=3))
        for i in range(nconv):

            model.add(
                tf.keras.layers.Conv2D(
                    filters=trial.suggest_int("filters{}".format(i), 32,
                                              self.max_conv_filters),
                    kernel_size=trial.suggest_categorical(
                        "kernel_size{}".format(i), [3, 5]),
                    strides=trial.suggest_categorical("strides{}".format(i),
                                                      [1, 2]),
                    padding="same",
                    activation=trial.suggest_categorical(
                        "activation{}".format(i), ["relu", "linear", "tanh"]),
                ))
            model.add(tf.keras.layers.BatchNormalization())
        dropout_rate = trial.suggest_float("Dropout", 0, 0.5)
        model.add(tf.keras.layers.Flatten())
        n_fully_con_layer = trial.suggest_categorical(
            "n_fully_con_layer", [1, self.max_fully_conn_layers])
        for i in range(n_fully_con_layer):
            units_fcl = trial.suggest_int("units_fcl{}".format(i), 200,
                                          self.max_units_fcl)
            model.add(tf.keras.layers.Dense(units_fcl, activation="relu"))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(self.classes, activation="softmax"))

        # Compile model.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer_name = trial.suggest_categorical("optimizer", [Adam, RMSprop, SGD])
        model.compile(
            optimizer=optimizer_name(learning_rate=lr),
            loss="categorical_crossentropy",  #list of loss suggestions
            metrics=["accuracy"],
        )

        return model

    def objective(self, trial):
        """Creating the objective function for ANN model optimization.

        Args:
            trial : optuna trial
        Returns:
            calculated loss
        """
        # Clear clutter from previous TensorFlow graphs.
        tf.keras.backend.clear_session()

        # Metrics to be monitored by Optuna.
        if tf.__version__ >= "2":
            monitor = "val_loss"
        else:
            monitor = "val_loss"

        # Create tf.keras model instance.
        model = self.create_model(trial)

        # Create callbacks for early stopping and pruning.
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3),
            TFKerasPruningCallback(trial, monitor),
        ]

        # Train model.
        history = model.fit(self.ds_train,
                            self.target,
                            batch_size=8,
                            epochs=self.EPOCHS,
                            verbose=True,
                            validation_split=0.2)
        print(monitor)

        return history.history[monitor][-1]

    def cnn_objective(self, trial):
        """Creating the objective function for ANN model optimization.

        Args:
            trial : optuna trial
        Returns:
            calculated loss
        """
        # Clear clutter from previous TensorFlow graphs.
        tf.keras.backend.clear_session()

        # Metrics to be monitored by Optuna.
        if tf.__version__ >= "2":
            monitor = "val_loss"
        else:
            monitor = "val_loss"

        # Create tf.keras model instance.
        model = self.cnn_model(trial)
        print("\n\n\n", model.summary())
        # Create callbacks for early stopping and pruning.
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3),
            TFKerasPruningCallback(trial, monitor),
        ]

        # Train model.
        history = model.fit(
            self.ds_train,
            self.target,
            epochs=self.EPOCHS,
            validation_split=0.1,
            steps_per_epoch=self.steps_per_epoch,
            batch_size=8,
        )

        return history.history[monitor][-1]

    def get_best_parameters(self, n_trials):
        """Gets the best parameter and the best model architecture.

        Args:
             n_trials : number of trials

        Returns:
            best_trial : the best trial.
            best_paramteres : the best parameters got after optimization.
            best_value : the best loss value corresponding to the trial.
        """
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2))

        a = datetime.datetime.now()
        if self.arch:
            study.optimize(self.cnn_objective, n_trials=n_trials, timeout=1800)
        else:
            study.optimize(self.objective, n_trials=n_trials, timeout=1800)
        print("\n\n\n\n Time taken to complete  trials",
              datetime.datetime.now() - a)
        pruned_trials = study.get_trials(deepcopy=False,
                                         states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False,
                                           states=[TrialState.COMPLETE])
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        return study.best_trial, study.best_params, study.best_value
