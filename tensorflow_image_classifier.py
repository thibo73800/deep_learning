
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Import Fashion MNIST
fashion_mnist = input_data.read_data_sets('input/data', one_hot=True)
# Label for Fashion MNIST
label_dict = {
0: "T-shirt/top",
1: "Trouser",
2: "Pullover",
3: "Dress",
4: "Coat",
5: "Sandal",
6: "Shirt",
7: "Sneaker",
8: "Bag",
9: "Ankle boot"
}

# Print the shape of the dataset (images)
print(fashion_mnist.train.images.shape)
# Print the shape of the dataset (targets)
print(fashion_mnist.train.labels.shape)

# Plot one example of image
plt.imshow(fashion_mnist.train.images[0].reshape(28, 28), cmap="gray")
plt.title(label_dict[np.argmax(fashion_mnist.train.labels[0])])
plt.show()

# Create the graph
# Placeholder
tf_features = tf.placeholder(tf.float32, [None, 784])
tf_targets = tf.placeholder(tf.float32, [None, 10])

# Variables
w1 = tf.Variable(tf.zeros([784, 10]))
b1 = tf.Variable(tf.zeros([10]))

# Neural network operations
z1 = tf.matmul(tf_features, w1) + b1
py = tf.nn.softmax(z1)

# Error and training
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf_targets, logits=z1)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Metrics: Accuracy
correct_prediction = tf.equal(tf.argmax(py,1), tf.argmax(tf_targets,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # Init variables
    sess.run(tf.global_variables_initializer())

    # Train the neural network
    for _ in range(1000):
        batch_xs, batch_ys = fashion_mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={tf_features: batch_xs, tf_targets: batch_ys})

    # Print the accuracy on the test set
    print("accuracy", sess.run(accuracy, feed_dict={tf_features: fashion_mnist.test.images, tf_targets: fashion_mnist.test.labels}))
