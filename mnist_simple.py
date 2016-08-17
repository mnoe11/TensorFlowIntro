# Import MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# Placeholder value for future TensorFlow inputs (of type flattened 28x28 image (784 dim vector))
x = tf.placeholder(tf.float32, [None, 784])

# Variables are persisted across all runs of the data graph and 
# may be modified across runs 
# weights: 784x10 matrix - 784 = 28x28 flattened numbers, 10 = possible classifications
# biases: 10x1 matrix - add to output of weights * inputs 
weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))

# Define the model as x (inputs) times weights + biases
y = tf.nn.softmax(tf.matmul(x, weights) + biases)

# Implements a cross-entropy function:
# http://colah.github.io/posts/2015-09-Visual-Information/
y_ = tf.placeholder(tf.float32, [None, 10])
# reduction_indices = 1 means that reduce_sum 
#  	(adds all elements, basically map + sum)
# is applied to the second dimention of y
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Specifies the training step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Initializes all variables, but does not run
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# Runs training on batches of 100
for i in range(1000):
	images, labels = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: images, y_: labels})

# Compares our predictions (y) to the actual labels (y_)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))