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

model = tf.keras.Sequential()

#must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

#Take a look at model summary
model.summary()

#Compile the model
model.compile(loss='categorical_crossentropy',optimizer ='adam', metrics=['accuracy'])

#Train the model
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5',verbose=1, save_best_only=True)
model.fit(x_train,y_train,batch_size=64,epochs=500, validation_data=(x_valid,y_valid), callbacks=[checkpointer])

#Load model with the best validation accuracy
model.load_weights('model.weights.best.hdf5')

# Test Accuracy
score=model.evaluate(x_test,y_test,verbose=0)

#prnt the test accuracy
print('Test Accuracy: ', score[1])

#Visualize prediction
y_hat=model.predict(x_test)

#Plat a random sample of 10 test images, their predicted labels and ground truth
figure=plt.figure(figsize=(20,8))
for i, index in enumerate(np.random.choice(x_test.shape[0],size=15,replace=False)):
	ax=figure.add_subplot(3,5,i+1,xticks=[],yticks=[])
	#display each image
	ax.imshow(np.squeeze(x_test[index]))
	predict_index=np.argmax(y_hat[index])
	true_index = np.argmax(y_test[index])

	#set the title for each image
	ax.set_title("{} ({})".format(fashion_mnist_label[predict_index],fashion_mnist_label[true_index]),color=("green" if predict_index == true_index else "red"))