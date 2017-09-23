
if __name__ == '__main__':
    # Function to minimise
    fc = lambda x, y: (3 * x**2) + (x * y) + (5 * y**2)
    # Set partial derivates
    partial_derivative_x = lambda x, y: (6 * x) + y
    partial_derivative_y = lambda x, y: (10 * y) + x
    # Set variables
    x = 10
    y = -13
    # Learning rate
    learning_rate = 0.01
    print("Fc = %s" % (fc(x, y)))
    # One epoch is one period of minimisation
    for epoch in range(0, 20):
        # Compute gradients
        x_gradient = partial_derivative_x(x, y)
        y_gradient = partial_derivative_y(x, y)
        # Apply gradient descent
        x = x - learning_rate * x_gradient
        y = y - learning_rate * y_gradient
        # Keep track of the function value
        print("Fc = %s" % (fc(x, y)))
    # Print final variables values
    print("")
    print("x = %s" % x)
    print("y = %s" % y)
