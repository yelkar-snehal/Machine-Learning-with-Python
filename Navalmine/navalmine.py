import numpy as np; #preprocessing
import pandas as pd; #dataframes
import tensorflow as tf; #for implementing dl
import matplotlib.pyplot as plt; #graphs
from sklearn.utils import shuffle; #preprocessing
from sklearn.preprocessing import LabelEncoder;
from sklearn.model_selection import train_test_split;


def read_dataset():
	#DF - DATAFRAME -- from sries dataframe and panel in pandas
	#read whole sheet
	df = pd.read_csv("sonar.csv");
	
	#features in X
	X = df[df.columns[0:60]].values;
	#
	
	#label in Y
	y = df[df.columns[60]];
	
	#encode R and M to digits 1 or 0 
	encoder = LabelEncoder();
	
	encoder.fit(y);
	y = encoder.transform(y);
	Y = one_hot_encoder(y);# hot coder to add 1 extra column to op
	#ie no cols present = no of labels	
	
	return (X, Y) ;

##############################################

def one_hot_encoder(labels):
	
	n_labels = len(labels);
	n_unique_labels	= len(np.unique(labels))
	one_hot_encode = np.zeros((n_labels, n_unique_labels));
	one_hot_encode[np.arange(n_labels), labels] = 1;
	return one_hot_encode;
	
#############################################

def multilayer_perceptron(x, weights, biases):
	# use relu for better convergence and handle vanishing gradients and sigmoid to balance activation
	
	#hidden layer relu
	#1st layer performs mat mult of ips and weights
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1']);
	layer_1 = tf.nn.relu(layer_1);
	
	#hidden layer sigmoid
	#2nd layer performs mat mult of layer1 and weights
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']);
	layer_2 = tf.nn.sigmoid(layer_2);
	
	#hidden layer sigmoid
	#3rd layer performs mat mult of layer2 and weights
	layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']);
	layer_3 = tf.nn.sigmoid(layer_3);
	
	#hidden layer relu
	#4th layer performs mat mult of layer3 and weights
	layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']);
	layer_4 = tf.nn.relu(layer_4);
	
	#op layer w/ linear activations
	out_layer = tf.matmul(layer_4, weights['out']) + biases['out'];
	
	return out_layer;


def main():
	
	#read the data set
	X, Y = read_dataset();
	
	# datatset -- 30% testing and 70% training splitting in sklearn
	train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.30, random_state = 42);
	
	#changes for each iteration
	learning_rate = 0.3;
	
	#no of of iterations 
	#greater the btter
	#however becomes stagnant after a point
	#iterat till that point  
	training_epochs = 1000;
	
	#array to store cost values in successive epochs
	cost_history = np.empty(shape = [1], dtype = float);
	
	#no of features = no of columns [1] tells dimensions of 1st row
	n_dim = X.shape[1];
	
	# 2 classes rock and mine r and m
	n_class = 2;
	
	#path for model files
	model_path = "demo_naval";
	
	# no of hiddden layeers and neurons presnet in each layer 
	#mostly neuron count is constant throughout 
	n_hidden_1 = 60;
	n_hidden_2 = 60;
	n_hidden_3 = 60;
	n_hidden_4 = 60;
	
	#placeholder to store ip 
	x = tf.compat.v1.placeholder(tf.float32, [None, n_dim]);
	
	#placeholder to store op
	y_ = tf.compat.v1.placeholder(tf.float32, [None, n_class]);
	
	# model params
	W = tf.Variable(tf.zeros([n_dim, n_class]));
	b = tf.Variable(tf.zeros([n_class]));
	
	#weights and biases
	
	#creating var w/ random values for weights
	weights = {
		'h1': tf.Variable(tf.random.truncated_normal([n_dim, n_hidden_1])),
		'h2': tf.Variable(tf.random.truncated_normal([n_hidden_1, n_hidden_2])),
		'h3': tf.Variable(tf.random.truncated_normal([n_hidden_2, n_hidden_3])),
		'h4': tf.Variable(tf.random.truncated_normal([n_hidden_3, n_hidden_4])),
		'out': tf.Variable(tf.random.truncated_normal([n_hidden_4, n_class])),
	}
	
	#creating var w/ random values for biases
	biases = {
		'b1': tf.Variable(tf.random.truncated_normal([n_hidden_1])),
		'b2': tf.Variable(tf.random.truncated_normal([n_hidden_2])),
		'b3': tf.Variable(tf.random.truncated_normal([n_hidden_3])),
		'b4': tf.Variable(tf.random.truncated_normal([n_hidden_4])),
		'out': tf.Variable(tf.random.truncated_normal([n_class])),
	}
	
	#init var
	init = tf.compat.v1.global_variables_initializer();
	
	saver = tf.compat.v1.train.Saver();
	
	#call to model fn for training
	y = multilayer_perceptron(x, weights, biases);
	
	#calculate avg loss
	#shape of  and y_ will be same w/ possibily diff dimensions 
	#tensor ie an array w/ softmax cross entropy loss given to reduce mean to get avg loss over the whole dataset
	cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y, labels = y_));
	
	#rduce loss byv gradient descent
	#minimise modifies w and b here
	training_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost_function);
	
	#launch graph
	sess = tf.compat.v1.Session();
	#necessary to get results from session 
	sess.run(init);
	#sess.run(tf.global_variables_initializer());
	accuracy_history = [];
	mse_history = [];
	
	#iterate fr no of epochs
	for epoch in range(training_epochs):
		
		sess.run(training_step, feed_dict = {x:train_x, y_:train_y});
		cost = sess.run(cost_function, feed_dict = {x:train_x, y_:train_y});
		cost_history = np.append(cost_history, cost);
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1));
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
		pred_y = sess.run(y, feed_dict = {x:test_x});
		mse = tf.reduce_mean(tf.square(pred_y - test_y));
		mse_ = sess.run(mse);
		accuracy = (sess.run(accuracy, feed_dict = {x:train_x, y_:train_y}));
		accuracy_history.append(accuracy);
		print('epoch: ', epoch, '-', 'cost: ',cost, "- MSE: ", mse_, "- Train Accuracy: ", accuracy);
	
	#model is saved to the file
	save_path = saver.save(sess, model_path)
	print("model saved in the file: %s ", save_path);
	
	#display graph for accuracy history
	plt.plot(accuracy_history);
	plt.title("acc history");
	plt.xlabel('Epoch');
	plt.ylabel('Acc');
	plt.show();
	
	#graph for loss calc
	plt.plot(range(len(cost_history)), cost_history);
	plt.title("loss calc");
	plt.axis([0,training_epochs, 0, np.max(cost_history)/100]);
	plt.xlabel('Epoch');
	plt.ylabel('loss');
	plt.show();
	
	#print final acc
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1));
	accuracy = tf.reduce_mean(tf.square(pred_y - test_y));
	print("test acc: ", (sess.run(y, feed_dict= {x:test_x, y_:test_y})));
	
	#print final mse
	pred_y = sess.run(y, feed_dict={x:test_x});
	mse = tf.reduce_mean(tf.square(pred_y -test_y));
	print("mse: ", sess.run(mse));			
		


###################################################################################################################
# entry pt fn	
if __name__ == "__main__":
	main();
