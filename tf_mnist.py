import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data

#Import Fashion MNIST
fashion_mnist = input_data.read_data_sets('input/data', one_hot=True)