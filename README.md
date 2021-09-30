# autoprototype

This library is designed for optimization of hyper-parameters in Deep Learning and Machine Learning Models. This module aims at
giving the user the best set of hyper-parameters for their DL/ML models. 
Along with suggesting the hyper-parameters, this module
aims at returning to the user the best suggestive model framework/architecture based on the user input data. The user has 
to just provide the dataset and the module will return :


1. The best model architecture based on the input data.
2. The best hyper-parameters for the model architecture.

In this way, the whole process of experimenting with the architecture is narrowed down. This is easy and requires only few lines of codes. 
Apart from the input data, only few other parameters are required by the module. All, the processes in rapid prototyping is automated thereafter, through this module.
The structure is based on some default values for parameters spaces required for the optimization. However, user would also have freedom to 
dynamically construct the search spaces for the hyperparameters.


This module is a wrap around the popular Hyper-parameter Optimization Tool called [`Optuna`](https://optuna.org/).
Optuna is used for the optimization process using iterative trails. This module takes the data as the primary input and 
suggests the user the model based on this Optuna trials. Optuna enables efficient hyperparameter optimization by adopting state-of-the-art
algorithms for sampling hyper-parameters and pruning efficiently unpromising trials. 

Some key features of Optuna, that we used are:

1.[Lightweight, versatile, and platform agnostic architecture](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html)

2. [Pythonic search spaces](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html)

3. [Efficient Optimization Algorithms](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)

This module, includiong all examples, is entirely written on Python.

## Installation Guide
Users can use pip install to install the packages to start working on the prototyping.
Below is given the pip command.

`pip install autoprototype`

The usage or the API calls for each supported ML/DL libraries are shown in the next sections.

#Supported Integrations Libraries

##SKLearn
At this point, the following models and their respective hyper-parameters are supported 

a. Decision Tree
        
b. SVC

c. Linear Regression

d. Ridge and Lasso Regression

e. Random Forest 

f. Logistic Regression

Starting the prototyping process consists of just two line of code 

        from autoprototype.sklearn import sklearnopt
        hpo = sklearnopt(X_train,y_train)
        trial , params , value = hpo.get_best_params(n_trials=trials)

This would return to us, the best trail, the set of best parameters including the model and the best objective value based on which the optimization is done.

To run the example navigate to [examples](https://github.com/Ideas2IT/auto-prototype/tree/master/examples) and use:

`python iris.py`


## tf.keras
At this point this supports, the following models and their hyper-parameters are supported:

##### Artificial Neural Networks

This also requires few lines of codes as shown below.

        from autoprototype.tf_keras import kerasopt
        hpo = kerasopt(x_train,y_train,EPOCHS=10,classes=CLASSES)
        trial , params , value = hpo.get_best_params(n_trials=n_trials)

By default it would run the trials and suggest you:

a. `n_layers` : Number of hidden layers in the model.

b. `units_i` : Number of units in each layers

c. `dropout_rate` : the dropout rate

d. `lr` : Learning rate of optimizers

e. `optimizer_name` : Name of the best optimizer

The loss criterion and the maximum number of layers, is set to `sparse_categorical_crossentropy` and `5` respectively, by default.
User can also provide any other loss function based on requirement as follows:

    hpo = HyperparamOpt(x_train,y_train,EPOCHS=10,classes=CLASSES,loss="your_loss_function",n_ann_layers=number_of layers)


To run the ANN example, navigate to [examples](https://github.com/Ideas2IT/auto-prototype/tree/master/examples) and run :

`python ann_tfkeras.py`

#### Convolution Neural Network

The API for CNN model is fairly the same as that of the above ANN. The user, must pass few other optional parameters to construct the suggestive CNN architecture. All other syntaxes are same.

        hpo = kerasopt(x_train,y_train,EPOCHS=10,classes=120,
                       max_units_fcl=400, max_conv_filters=1000,
                       arch="cnn",input_shape=(128,128,3),steps_per_epoch=10)

Two mandatory arguments to run CNN:

`arch` refers to the specification of architecture to 'cnn', 

`input_shape` must be provided for any CNN models.

Other optional arguments


`max_units_fcl` : max units in the fully connected layers of CNN, default `1000`

`max_nconv` : max number of convolution layers, default `5`

`max_fully_conn_layers` : max number of fully connected layers. default `2`

`steps_per_epoch` : Number of steps to be taken in an epoch, default `216`

`max_conv_filters` : maximum filters in the convolution layers, default `256`


To run the CNN example [tf_keras/examples] and run :

Download the data from :

`https://www.kaggle.com/c/dog-breed-identification/data`

Navigate to examples folder

`cd examples`
Make an empty directory called data. 
`mkdir data`
move the downloaded data into the `data` folder. Please check the path to data provided in the `cnn_tfkeras.py` script!

Run:
`python cnn_tfkeras.py`
