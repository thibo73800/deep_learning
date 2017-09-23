import matplotlib.pyplot as plt
import numpy as np


def activation(z):
    """
        Activation/Sigmoid
        **input: **
            *z: (Integer|Numpy Array)
    """
    return 1 / (1 + np.exp(-z))


def derivative_activation(z):
    """
        Derivative of the activation/Sigmoid
        **input: **
            *z: (Integer|Numpy Array)
    """
    return activation(z) * (1 - activation(z))


def pre_activation(features, weights, bias):
    """
        Compute the pre activation
        **input: **
            *features: (Numpy Matrix)
            *weights: (Numpy vector)
            *bias: (Integer)
    """
    return np.dot(features, weights) + bias


def init_variables():
    """
        Init model variables (weights and bias)
    """
    weights = np.random.normal(size=2)
    bias = 0

    return weights, bias


def predict(features, weights, bias):
    """
        Predict the class
        **input: **
            *features: (Numpy Matrix)
            *weights: (Numpy vector)
            *bias: (Integer)
        **reutrn: (Numpy vector)**
            *0 or 1
    """
    z = pre_activation(features, weights, bias)
    y = activation(z)
    return np.round(y)


def get_dataset():
    """
        Method used to generate the dataset
    """
    # Numbers of row per class
    row_per_class = 100
    # Generate rows
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    healthy = np.random.randn(row_per_class, 2) + np.array([2, 2])

    features = np.vstack([sick, healthy])
    targets = np.concatenate((np.zeros(row_per_class), np.zeros(row_per_class) + 1))

    return features, targets


def cost(predictions, targets):
    """
        Compute the cost of the model
        **input: **
            *predictions: (Numpy vector) y
            *targets: (Numpy vector) t
    """
    return np.mean((predictions - targets) ** 2)


def train(features, targets, weights, bias):
    """
        Method used to train the model using the gradient descent method
        **input: **
            *features: (Numpy Matrix)
            *targets: (Numpy vector)
            *weights: (Numpy vector)
            *bias: (Integer)
        **return (Numpy vector, Numpy vector) **
            *update weights
            *update bias
    """
    epochs = 100
    learning_rate = 0.1

    # Print current Accuracy
    predictions = predict(features, weights, bias)
    print("Accuracy = %s" % np.mean(predictions == targets))

    # Plot points
    plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
    plt.show()

    for epoch in range(epochs):
        # Compute and display the cost every 10 epoch
        if epoch % 10 == 0:
            predictions = activation(pre_activation(features, weights, bias))
            print("Current cost = %s" % cost(predictions, targets))
        # Init gragients
        weights_gradients = np.zeros(weights.shape)
        bias_gradient = 0.
        # Go through each row
        for feature, target in zip(features, targets):
            # Compute prediction
            z = pre_activation(feature, weights, bias)
            y = activation(z)
            # Update gradients
            weights_gradients += (y - target) * derivative_activation(z) * feature
            bias_gradient += (y - target) * derivative_activation(z)
        # Update variables
        weights = weights - (learning_rate * weights_gradients)
        bias = bias - (learning_rate * bias_gradient)
    # Print current Accuracy
    predictions = predict(features, weights, bias)
    print("Accuracy = %s" % np.mean(predictions == targets))


if __name__ == '__main__':
    # Dataset
    features, targets = get_dataset()
    # Variables
    weights, bias = init_variables()
    # Train the model
    train(features, targets, weights, bias)
