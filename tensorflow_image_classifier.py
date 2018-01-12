import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("input/data", one_hot=True)

# Graph Inputs
tf_features = tf.placeholder(tf.float32, [None, 784])
tf_targets = tf.placeholder(tf.float32, [None, 10])

# Variables
w1 = tf.Variable(tf.random_normal([784, 10]))
b1 = tf.Variable(tf.zeros([10]))

#  Operations
z1 = tf.matmul(tf_features, w1) + b1
softmax = tf.nn.softmax(z1)

# Error + Train
error = tf.nn.softmax_cross_entropy_with_logits(labels=tf_targets, logits=z1)
train = tf.train.GradientDescentOptimizer(0.5).minimize(error)

# Metrics: Accuracy
correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(tf_targets, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(mnist.train.labels[0])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Train the model
    epochs = 1000
    for e in range(epochs):
        batch_features, batch_targets = mnist.train.next_batch(100)
        sess.run(train, feed_dict={tf_features: batch_features, tf_targets: batch_targets})

    true_cls = []
    py_cls = []
    # Print the prediction for the first 10 predictions
    for c in range(0, 10):
        py = sess.run(softmax, feed_dict={
            tf_features: [mnist.test.images[c]]
        })
        true_cls.append(np.argmax(mnist.test.labels[c]))
        py_cls.append(np.argmax(py))
    print("true_cls", true_cls)
    print("py cls", py_cls)

    # Accuracy on the test set
    acc = sess.run(accuracy, feed_dict={
        tf_features: mnist.test.images,
        tf_targets:  mnist.test.labels
    })
    print("Accuracy on the test set", acc)
