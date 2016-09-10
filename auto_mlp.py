import itertools

import numpy as np
import GPyOpt

from data_preprocessing import norm_data, stand_data, norm_data_reverse, \
    stand_data_reverse, train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU

import matplotlib.pyplot as plt


class MLP:
    def __init__(self, nb_hidden_layers, input_layer_neurons,
                 hidden_layer_neurons, activation_type, dropout_rate):
        """ Initialize a MLP model specified by the given hyperparameters.

        :param int nb_hidden_layers: How many hidden layers does the MLP have
        :param int input_layer_neurons:
        :param int hidden_layer_neurons:
        :param int activation_type: Which activation function is selected
        :param float dropout_rate: The level of Dropout after input layer and
        each hidden layer.
        """
        # Keras expects integers while GPyOpt defaults to floats
        self.nb_hidden_layers = int(nb_hidden_layers)
        self.input_layer_neurons = int(input_layer_neurons)
        self.hidden_layer_neurons = int(hidden_layer_neurons)
        self.activation_type = int(activation_type)
        self.dropout_rate = dropout_rate
        self.model = None

    def build_model(self, input_dim):
        """ Builds a MLP model specified by the hyperparameters and an input
        dimension.

        :param int input_dim: The dimensionality of the input data.
        """
        model = Sequential()

        # input, hidden, and output layers
        total_layers = self.nb_hidden_layers+2

        for i in range(total_layers):
            if i == 0:  # input layer
                model.add(Dense(self.input_layer_neurons,
                                input_dim=input_dim))
            elif i == total_layers-1:  # output layer
                model.add(Dense(1))
            else:   # hidden layers
                model.add(Dense(self.hidden_layer_neurons))

            if i < total_layers - 1:    # input and hidden layers
                # modify neuronal properties based on the hyperparameter values
                if self.activation_type == 0:
                    model.add(Activation('relu'))
                elif self.activation_type == 1:
                    model.add(PReLU())

                # add a Dropout layer to avoid overfitting
                model.add(Dropout(self.dropout_rate))

        self.model = model

    def train_model(self, X_train, y_train, batch_size, nb_epoch):
        """ Train the model with input and output training data. The given
        minibatch size is used for a given number of epochs.

        :param 2d np.ndarray X_train: Training input data with dimensions
        examples x input dimension.
        :param 2d np.ndarray y_train: Training output data with dimensions
        examples x 1.
        :param int batch_size: Minibatch size.
        :param int nb_epoch: Number of training epochs.
        """
        self.model.compile(loss="mse", optimizer='rmsprop')
        self.model.fit(X_train, y_train, batch_size=batch_size,
                       nb_epoch=nb_epoch, verbose=1)

    def predict(self, X_test, target, postprocess_func=None):
        """ Predict using a trained model. If a postprocessing function is
        given, then both the predictions and target values are scaled using it.

        :param 2d np.ndarray X_test: Test input data with dimensions examples x
        input dimension.
        :param 2d np.array target: Test output data with dimensions examples x
        1.
        :param function postprocess_func: An optional postprocessing function,
        which is applied to both predictions and targets.
        """
        prediction = self.model.predict(X_test, verbose=0)

        if postprocess_func is not None:
            prediction = postprocess_func(prediction)
            target = postprocess_func(target)

        return prediction, target

    def evaluate_mse(self, prediction, target):
        """ Evaluate the mean squared error of a model.
        :param np.ndarray prediction:
        :param np.ndarray target:
        """
        mse = sum((target-prediction)**2)

        return mse


class MLPOptimizer:
    def __init__(self, X_train, y_train, X_test, y_test, postprocess_funcs,
                 batch_size, nb_epoch):
        """
        Initialize data required for finding the optimal data preprocessing
        and MLP architecture.

        :param 3d np.ndarray X_train: Training input data. The first dimension
        contains different preprocessing types, the second dimension has
        examples and the third dimension is the input dimension.
        :param 3d np.ndarray y_train: Ditto.
        :param 3d np.ndarray X_test: Ditto but test input data.
        :param 3d np.ndarray y_test: Ditto.
        :param list[function] postprocess_funcs: A list of functions which can
        postprocess data output by a model.
        :param int batch_size: The minibatch size for the gradient descent.
        :param int nb_epoch: Number of training epochs. 
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.postprocess_funcs = postprocess_funcs
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

    def f(self, X):
        """ The GPyOpt library requires function f, which evaluates the function
        being optimized at a given point X and returns a loss.

        :param list X: The point at which the function f is evaluated at.
        :return float loss: Loss measure
        """
        # GPyOpt adds an extra dimension and uses float by default
        preprocess_type = int(X[0][0])
        hidden_layers = int(X[0][1])
        input_layer_neurons = int(X[0][2])
        hidden_layer_neurons = int(X[0][3])
        activation_type = int(X[0][4])
        dropout_rate = X[0][5]

        input_dim = self.X_train[0].shape[1]

        # print the choices
        print('preprocess type', preprocess_type)
        print('hidden layers', hidden_layers)
        print('input and hidden layer neurons', input_layer_neurons,
              hidden_layer_neurons)
        print('activation type', activation_type)
        print('dropout rate', dropout_rate)

        # initialize an MLP model
        mlp = MLP(hidden_layers, input_layer_neurons, hidden_layer_neurons,
                  activation_type, dropout_rate)
        mlp.build_model(input_dim)

        X_train = self.X_train[preprocess_type]
        y_train = self.y_train[preprocess_type]
        X_test = self.X_test[preprocess_type]
        y_test = self.y_test[preprocess_type]

        mlp.train_model(X_train, y_train, self.batch_size, self.nb_epoch)

        postprocess_func = self.postprocess_funcs[preprocess_type]
        prediction, target = mlp.predict(X_test, y_test, postprocess_func)
        loss = mlp.evaluate_mse(prediction, target)

        return loss


def generate_model_alternatives():
    # specify sensible ranges/alternatives for each hyperparameter
    preprocess_type = [0, 1]
    hidden_layers = [1, 2, 3, 4, 5]
    input_layer_neurons = [15, 30, 60, 120, 240, 480]
    hidden_layer_neurons = [10, 20, 40, 60, 120, 240]
    activation_type = [0, 1]
    dropout_rate = [0.1, 0.2, 0.3, 0.4]

    # produce the cartesian product of the different hyperparameters
    alternatives = itertools.product(preprocess_type, hidden_layers,
                                     input_layer_neurons, hidden_layer_neurons,
                                     activation_type, dropout_rate)

    return list(alternatives)


def main():
    # read input and output data
    X = np.loadtxt('datasets/fi_price/input.csv', delimiter=",", skiprows=1)
    y = np.loadtxt('datasets/fi_price/output.csv', delimiter=",", skiprows=1)

    X = X[:, 2:]     # skip timestamp columns

    # reshape y to a two-dimensional array
    y = y.reshape(y.shape[0], 1)

    # save output statistics for scaling back to absolute values
    y_std = y.std(axis=0)
    y_mean = y.mean(axis=0)
    y_max = y.max(axis=0)
    y_min = y.min(axis=0)

    # preprocess data
    X_stand = stand_data(X)     # standardization
    y_stand = stand_data(y)

    norm_lb = 0     # normalization to the interval [norm_lb, norm_ub]
    norm_ub = 1
    X_norm = norm_data(X, norm_lb, norm_ub)
    y_norm = norm_data(y, norm_lb, norm_ub)

    # fix parameters beforehand for scaling back to absolute figures
    postprocess_funcs = \
        [lambda x: stand_data_reverse(x, y_mean, y_std),
         lambda x: norm_data_reverse(x, norm_lb, norm_ub, y_min, y_max)]

    # split the data to train and test sets and cache the results
    (X_train_stand, y_train_stand), (X_test_stand, y_test_stand) = \
        train_test_split(X_stand, y_stand)
    (X_train_norm, y_train_norm), (X_test_norm, y_test_norm) = \
        train_test_split(X_norm, y_norm)

    X_train = [X_train_stand, X_train_norm]
    y_train = [y_train_stand, y_train_norm]
    X_test = [X_test_stand, X_test_norm]
    y_test = [y_test_stand, y_test_norm]

    # model hyperparameters which are not optimized
    nb_epoch = 100
    batch_size = 500

    # setup the model type to be optimized
    mlp_optimizer = MLPOptimizer(X_train, y_train, X_test, y_test,
                                 postprocess_funcs, batch_size, nb_epoch)

    # setup the domain of the hyperparameters
    model_alternatives = generate_model_alternatives()
    print('The number of model alternatives is', len(model_alternatives))
    domain = [{'name': 'test', 'type': 'bandit', 'domain': model_alternatives}]

    # set up the Bayesian optimization instance
    # function evaluations are not exact because training is stochastic
    bayes_opt = GPyOpt.methods.BayesianOptimization(
        f=mlp_optimizer.f, domain=domain, acquisition_type='LCB',
        exact_feval=False, initial_design_numdata=10)

    # run Bayesian optimization for a specified number of iterations
    max_iter = 10
    max_time = 10*60*60    # in seconds
    bayes_opt.run_optimization(max_iter, max_time)

    # plot GP and convergence statistics
    bayes_opt.plot_convergence()

    # obtain the best preprocessing method and hyperparameters
    print('Best preprocessing and hyperparameter configuration',
          bayes_opt.x_opt)
    opt_preprocess_type = int(bayes_opt.x_opt[0])
    opt_hyperparameters = bayes_opt.x_opt[1:]

    print('Initializing the model with the best hyperparameters')
    opt_mlp = MLP(*opt_hyperparameters)

    X_train = X_train[opt_preprocess_type]
    y_train = y_train[opt_preprocess_type]

    opt_mlp.build_model(X_train.shape[1])
    opt_mlp.train_model(X_train, y_train, batch_size, nb_epoch)

    # Predict using the optimized MLP model
    X_test = X_test[opt_preprocess_type]
    y_test = y_test[opt_preprocess_type]
    postprocess_func = postprocess_funcs[opt_preprocess_type]

    prediction, target = opt_mlp.predict(X_test, y_test, postprocess_func)

    # plot the prediction and the ground truth
    plt.figure()
    ts = np.arange(y_test.shape[0])
    plt.plot(ts, target, 'b', label='target')
    plt.plot(ts, prediction, 'r', label='prediction')

    plt.legend(loc='best')
    plt.xlabel('timestep')
    plt.ylabel('EUR/MWh')

    plt.show()

main()
