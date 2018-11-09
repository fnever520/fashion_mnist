import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("Shape of x_train: ", x_train.shape, "\nShape of y_train: ", y_train.shape)

# Print the number of training and test datasets
print(x_train.shape[0],"train set")
print(x_test.shape[0], "test set")

# Define the text labels
fashion_mnist_label = ["T-shirt",
						"Trouser",
						"Pullover",
						"Dress",
						"Coat",
						"Sandals",
						"Shirt",
						"Sneaker",
						"Bag",
						"Ankle boot"
]

#Specify your image number
img_index = 56666

#y_train contains the labels ranging from 0 to 9
label_index=y_train[img_index]

# Print the label 
print("y = " + str(label_index) + " " + (fashion_mnist_label[label_index]))

plt.imshow(x_train[img_index])

#Data normalization
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

print('Number of train data ' + str(len(x_train)))
print('Number of test data ' + str(len(x_test)))

#split the data into train/validation/test data sets

(x_train, x_valid) = x_train[5000:],x_train[:5000]
(y_train, y_valid) = y_train[5000:],y_train[:5000]


# Reshape the inptu data from (28,28) to (28,28,1)
width, height = 28,28
x_train = x_train.reshape(x_train.shape[0],width, height,1)
x_valid = x_valid.reshape(x_valid.shape[0],width, height,1)
x_test = x_test.reshape(x_test.shape[0],width, height,1)

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print(x_train.shape)
# Print the training set shape
print("x_train shape: ", x_train.shape, "\ny_train shape: ", y_train.shape)

#Print the number of training, validation and test datasets
print(x_train.shape[0],'train set')
print(x_valid.shape[0],'valid set')
print(x_test.shape[0],'test set')

model = tf.keras.Sequential