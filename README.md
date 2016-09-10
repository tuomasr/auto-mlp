# auto-mlp
The model automatically finds the optimal preprocessing method (e.g. normalization) and MLP hyperparameters (e.g. number of hidden layers and neurons in them) for a given dataset. Thus, one can employ an MLP model without any prior expertise. The model uses Bayesian optimization and depends on GPyOpt https://github.com/SheffieldML/GPyOpt

Here is a sample of an automatically fitted model to one of the provided datasets.
![Image of an automatically fitted model](https://github.com/tuomasr/auto-mlp/blob/master/sample.png)
