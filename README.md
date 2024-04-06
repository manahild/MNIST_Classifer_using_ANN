# MNIST Dataset Classification
This project aims to classify handwritten digits from the MNIST dataset using a simple neural network built with TensorFlow/Keras.

## Overview
The MNIST dataset is a classic benchmark dataset consisting of 28x28 grayscale images of handwritten digits (0 through 9) and their respective labels. The goal is to train a neural network model to accurately predict the digit represented in each image.

## Prerequisites
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

## Model Architecture
- The neural network model used in this project is a simple feedforward network with two dense layers.
 The architecture is as follows:
- Flatten Layer: Input layer to flatten the 28x28 image into a 1D array of size 784.
- Dense Layer (128 units, ReLU activation): Hidden layer with 128 neurons and ReLU activation function.
- Dense Layer (10 units, Softmax activation): 
- Output layer with 10 neurons representing the digit classes and softmax activation function.
## Training
The model is trained using the Adam optimizer and sparse categorical cross-entropy loss function. It is trained on the training set for 20 epochs with a batch size of 64.



