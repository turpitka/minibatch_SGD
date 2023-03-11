# minibatch_SGD
This repository contains implementation of a mini-batched Gradient Descent Classifier with the following loss functions: L2, count, and cross-entropy (see part 6) and the types of weight decay: L1, L2, none.

Usage
The function takes in the following parameters:

- X_train: training features
- y_train: training target values
- X_test: test features
- y_test: test target values
- weight_decay_factor: weight decay factor
- wd_type: form of weight decay (L1, L2, none)
- loss_f: type of loss function (L2, count, cross-entropy)
- learning_rate: learning rate
- batch_size: mini-batch size
- n_epochs: number of epochs

The function returns the optimized weights as well as plots the loss and accuracy functions.

## Implementation Details
The function implements stochastic gradient descent with mini-batches to minimize the loss and evaluate the train and test MSE.
The learning rate and weight decay factor are tuned based on the given parameters.
The train and test loss are shown as a function of epochs, where the number of epochs is chosen to ensure the train loss is minimized.

## Example

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set parameters
learning_rate = 0.01
batch_size = 10
n_epochs = 500
weight_decay_factor = 0.01
wd_type = 'L2'
loss_f = 'cross-entropy'

# Run minibatch_SGD function
weights = minibatch_SGD(X_train, y_train, X_test, y_test, weight_decay_factor, wd_type, loss_f, learning_rate, batch_size, n_epochs)
```

## Results
The function achieved an average accuracy of 75% on a benchmark Iris dataset using cross-entropy loss for softmax classification. 
It also achieved a near 50% accuracy in image classification on the Cifar-10 dataset with the cross-entropy loss.

## Future Work

In addition to the current accuracy calculation, it would be beneficial to implement the average accuracy per class in the evaluation metrics, 
and then take the average of those class accuracies to get the overall accuracy of the model.

