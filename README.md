# auto-mlp
The model automatically finds the optimal preprocessing method (e.g. normalization) and MLP hyperparameters (e.g. number of hidden layers and neurons in them) for a given dataset. Thus, one can provide a dataset and let the model figure out the optimal MLP model to it. The model uses Bayesian optimization and depends on GPyOpt https://github.com/SheffieldML/GPyOpt

Here is a sample of an automatically fitted model to one of the provided datasets. Better fit can be obtained by running the model for more iterations.
![Image of an automatically fitted model](https://github.com/tuomasr/auto-mlp/blob/master/sample.png)
