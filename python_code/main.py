# libraries
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import math
import time

# setting random seed
np.random.seed(77)
tf.set_random_seed(77)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# paths
train_data_path = "./../data/train_data.csv"
test_data_path = "./../data/test_data.csv"
target_tf_graph_path = "./../trained_model/tf_graph/"
target_tf_ckpt_path = "./../trained_model/tf_ckpt/model.ckpt"

# reading data set
train_data = pd.read_csv(train_data_path, header=None)
test_data = pd.read_csv(test_data_path, header=None)

# separating features and labels
train_X = train_data.iloc[:, :784]
train_Y = pd.get_dummies(train_data[784])
test_X = test_data.iloc[:, :784]
test_Y = np.array(test_data[784])
print("Train Data: " + str(train_X.shape))
print("Train Labels: " + str(train_Y.shape))
print("Test Data: " + str(test_X.shape))
print("Test Labels: " + str(test_Y.shape))

# tf Graph input
x = tf.placeholder("float", [None, n_input], name="x")
y = tf.placeholder("float", [None, n_classes], name="y")

# first layer
weight1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='weight1')
bias1 = tf.Variable(tf.random_normal([n_hidden_1]), name="bias1")
layer_1 = tf.add(tf.matmul(x, weight1), bias1, name="layer_1")
relu_layer_1 = tf.nn.relu(layer_1, name="relu_layer_1")

# second layer
weight2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="weight2")
bias2 = tf.Variable(tf.random_normal([n_hidden_2]), name="bias2")
layer_2 = tf.add(tf.matmul(relu_layer_1, weight2), bias2, name="layer_2")
relu_layer_2 = tf.nn.relu(layer_2, name="relu_layer_2")

# output layer
weight3 = tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name="weight3")
bias3 = tf.Variable(tf.random_normal([n_classes]), name="bias3")
out_layer = tf.add(tf.matmul(relu_layer_2, weight3), bias3, name="out_layer")

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y), name="cost")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer").minimize(cost)

# prediction
pred = tf.argmax(out_layer, 1, name="pred")	#DON'T DELETE -> , output_type=tf.int32)

# accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(test_Y, pred), "float"), name="accuracy")

# Initializing the variables
init = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')

# making tensorflow session
sess = tf.Session()

# initialzing session
sess.run(init)

start_time = time.time()
# training network
num_batches = int(math.ceil(len(train_X) / float(batch_size)))
for e in range(training_epochs):
	avg_cost = 0
	batch_index = 0
	while(batch_index < num_batches):
		# making train batches
		batch_X = train_X.iloc[batch_index * batch_size : (batch_index + 1) * batch_size, :]
		batch_Y = train_Y.iloc[batch_index * batch_size : (batch_index + 1) * batch_size, :]
		batch_index += 1
		# training network
		_, loss = sess.run([optimizer, cost], feed_dict={x: batch_X, y: batch_Y})
		avg_cost += loss
	avg_cost = avg_cost / float(len(train_X))
	print("Epoch: " + str(e) + " Loss: " + str(avg_cost))

# testing network
predictions = sess.run(pred, feed_dict={x: test_X})

# getting accuracy
acc = accuracy_score(test_Y, predictions)
tf_acc = sess.run(accuracy, feed_dict={x: test_X})
print("Test Accuracy: " + str(acc) + "   " + str(tf_acc))

# saving all trainable variables (weights and bias)
for variable in tf.trainable_variables():
	tensor = tf.constant(variable.eval(sess))
	tf.assign(variable, tensor, name='nWeights')
tf.train.write_graph(sess.graph_def, target_tf_graph_path, 'graph.pb', as_text=False)
saver = tf.train.Saver()
saver.save(sess, target_tf_ckpt_path)
