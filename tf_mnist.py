import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data

#Import Fashion MNIST with one hot encoding
fashion_mnist = input_data.read_data_sets('data/fashion', one_hot=True)
#fashion_mnist = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
label_dict = {
	0: 'T-Shirt',
	1: 'Trouser',
	2: 'Pullover',
	3: 'Dress',
	4: 'Coat',
	5: 'Sandal',
	6: 'Shirt',
	7: 'Sneaker',
	8: 'Bag',
	9: 'Ankle boot'
}

#Sample 1
'''
# Get 28*28 image
sample_1 = fashion_mnist.train.images[47].reshape(28,28)
#Get corresponding integer label from one-hot encoded data
sample_label_1 = np.where(fashion_mnist.train.labels[47] ==1)[0][0]

#plot sample
print("y={label_index} ({label})".format(label_index=sample_label_1, label=label_dict[sample_label_1]))
plt.title(label_dict[sample_label_1])
plt.imshow(sample_1,cmap='Greys')
plt.show()

# Get 28x28 image
sample_2 = fashion_mnist.train.images[23].reshape(28,28)
# Get corresponding integer label from one-hot encoded data
sample_label_2 = np.where(fashion_mnist.train.labels[23] == 1)[0][0]
# Plot sample
print("y = {label_index} ({label})".format(label_index=sample_label_2, label=label_dict[sample_label_2]))
plt.title(label_dict[sample_label_2])
plt.imshow(sample_2, cmap='Greys')
plt.show()
'''

# Network parameters
n_hidden_1 = 128 # Units in first hidden layer
n_hidden_2 = 128 # Units in second hidden layer
n_input = 784 # Fashion MNIST data input (img shape: 28*28)
n_classes = 10 # Fashion MNIST total classes (0–9 digits)
n_samples = fashion_mnist.train.num_examples # Number of examples in training set 

# Create placeholders
def create_placeholders(num_input, num_output):
 '''
 Creates the placeholders for the tensorflow session.
 
 Arguments:
 num_input -- scalar, size of an image vector (28*28 = 784)
 num_output -- scalar, number of classes (10)
 
 Returns:
 X -- placeholder for the data input, of shape [num_input, None] and dtype "float"
 Y -- placeholder for the input labels, of shape [num_output, None] and dtype "float"
 '''
 
 X = tf.placeholder(tf.float32, [num_input, None], name='X')
 Y = tf.placeholder(tf.float32, [num_output, None], name='Y')
 
 return X, Y

def initialize_parameters():
    '''
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [n_hidden_1, n_input]
                        b1 : [n_hidden_1, 1]
                        W2 : [n_hidden_2, n_hidden_1]
                        b2 : [n_hidden_2, 1]
                        W3 : [n_classes, n_hidden_2]
                        b3 : [n_classes, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    '''
    
    # Set random seed for reproducibility
    tf.set_random_seed(42)
    
    # Initialize weights and biases for each layer
    # First hidden layer
    # TRYME
    # Xavier initializiation --- xavier_initializer.
    # He initialization --- variance_scaling_initializer
    W1 = tf.get_variable("W1", shape=[n_hidden_1, n_input], initializer=tf.contrib.layers.variance_scaling_initializer(seed=42))
    b1 = tf.get_variable("b1", shape=[n_hidden_1, 1], initializer=tf.zeros_initializer())
    
    # Second hidden layer
    W2 = tf.get_variable("W2", shape=[n_hidden_2, n_hidden_1], initializer=tf.contrib.layers.variance_scaling_initializer(seed=42))
    b2 = tf.get_variable("b2", shape=[n_hidden_2, 1], initializer=tf.zeros_initializer())
    
    # Output layer
    W3 = tf.get_variable("W3", [n_classes, n_hidden_2], initializer=tf.contrib.layers.variance_scaling_initializer(seed=42))
    b3 = tf.get_variable("b3", [n_classes, 1], initializer=tf.zeros_initializer())
    
    # Store initializations as a dictionary of parameters
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }
    
    w_h_1 = tf.summary.histogram("weights",W1)
    b_h_1 = tf.summary.histogram("bias",b1)
    w_h_2 = tf.summary.histogram("weights",W2)
    b_h_2 = tf.summary.histogram("bias",b2)
    w_h_3 = tf.summary.histogram("weights",W3)
    b_h_3 = tf.summary.histogram("bias",b3)
    
    return parameters

def conv(X, parameters):
    '''
    Implements the forward propagation for the model: 
    CNN -> RELU -> CNN -> RELU -> CNN -> SOFTMAX
    
    Arguments:
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel

    # Convolutional Layer #1
    # Computes 32 features using a 2x2 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    '''

    # 1st CNN layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[2, 2],
        padding="same",
        activation=tf.nn.relu)    

	# Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 2nd CNN layer 
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[2, 2],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


    # Retrieve parameters from dictionary
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    # Carry out forward propagation   
    # TRYME: relu or swish   
    Z1 = tf.add(tf.matmul(W1,X), b1)     
    A1 = tf.nn.relu(Z1)                  
    Z2 = tf.add(tf.matmul(W2,A1), b2)    
    A2 = tf.nn.relu(Z2)                  
    Z3 = tf.add(tf.matmul(W3,A2), b3) 
    print("check the shape of Z3: ", Z3.shape)
    
    return Z3

def forward_propagation(X, parameters):
    '''
    Implements the forward propagation for the model: 
    LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    '''
    
    # Retrieve parameters from dictionary
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    # Carry out forward propagation   
    # TRYME: relu or swish   
    Z1 = tf.add(tf.matmul(W1,X), b1)     
    A1 = tf.nn.relu(Z1)                  
    Z2 = tf.add(tf.matmul(W2,A1), b2)    
    A2 = tf.nn.relu(Z2)                  
    Z3 = tf.add(tf.matmul(W3,A2), b3) 
    print("check the shape of Z3: #109", Z3.shape)
    
    return Z3

def compute_cost(Z3, Y):
    '''
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (10, number_of_examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    '''
    
    # Get logits (predictions) and labels
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    # Compute cost
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    with tf.name_scope("cost_function"):
    	tf.summary.scalar("cost_function", cost_function)
    
    return cost_function

def model(train, test, learning_rate=0.0001, num_epochs=100, sizeMiniBatch=32, print_cost=True, graph_filename='costs', dest = r"C:\tmp\tensorflow_logs"):
    '''
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    train -- training set
    test -- test set
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    sizeMiniBatch -- size of a minibatch
    print_cost -- True to print the cost every epoch
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    '''
    
    # Ensure that model can be rerun without overwriting tf variables
    ops.reset_default_graph()
    # For reproducibility
    tf.set_random_seed(42)
    seed = 42
    # Get input and output shapes
    (num_input, num_training) = train.images.T.shape
    num_output = train.labels.T.shape[0]
    
    costs = []
    
    # Create placeholders of shape (num_input, num_output)
    X, Y = create_placeholders(num_input, num_output)
    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation
    Z3 = forward_propagation(X, parameters)
    # Cost function
    cost_function = compute_cost(Z3, Y)
    
    # Backpropagation (using Adam optimizer)
    # TRYME: with AdamOptimizer or AdagradOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)
    
    # Initialize variables
    init = tf.global_variables_initializer()
    
    # Merge all the summaries
    merge_summary = tf.summary.merge_all()

    # Start session to compute Tensorflow graph
    with tf.Session() as sess:
        
        # Run initialization
        sess.run(init)
        summary_writer = tf.summary.FileWriter(dest, graph = sess.graph)
        counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            
            epoch_cost = 0.
            num_minibatches = int(num_training / sizeMiniBatch)
            seed = seed + 1
           
            for i in range(num_minibatches):
                
                # Get next batch of training data and labels
                minibatch_X, minibatch_Y = train.next_batch(sizeMiniBatch)
                counter += 1
                # Execute optimizer and cost function
                summary, _ , minibatch_cost = sess.run([ merge_summary, optimizer, cost_function], feed_dict={X: minibatch_X.T, Y: minibatch_Y.T})
                summary_writer.add_summary(summary,counter)
 
                # Update epoch cost
                epoch_cost += minibatch_cost / num_minibatches
                
            # Print the cost every epoch
            if print_cost == True:
                print("Cost after epoch {epoch_num}: {cost}".format(epoch_num=epoch, cost=epoch_cost))
                costs.append(epoch_cost)
        
        # Plot costs
        plt.figure(figsize=(16,5))
        plt.plot(np.squeeze(costs), color='#2A688B')
        plt.xlim(0, num_epochs-1)
        plt.ylabel("cost_function")
        plt.xlabel("iterations")
        plt.title("learning rate = {rate}".format(rate=learning_rate))
        plt.savefig(graph_filename, dpi=300)
        plt.show()
 
        # Save parameters
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        
        # Calculate correct predictions
        # Z3 is the predicted, and Y is the true label
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        
        # Calculate accuracy on test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        print ("[Train Accuracy\t]:", accuracy.eval({X: train.images.T, Y: train.labels.T}))
        print ("[Test Accurac\t]:", accuracy.eval({X: test.images.T, Y: test.labels.T}))
        
        return parameters

# Running the model
train = fashion_mnist.train
test = fashion_mnist.test

parameters = model(train, test, learning_rate=0.0001, num_epochs=100, sizeMiniBatch=32, dest = r"C:\tmp\tensorflow_logs\lr_1e-4_he_initializer")
