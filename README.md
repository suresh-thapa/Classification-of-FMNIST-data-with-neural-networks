# Classification-of-FMNIST-data-with-neural-networks

This work deals with the classification of Fashion MNIST dataset using artificial neural network in Python
TensorFlow. The dataset contains 60,000 images for training and validation and 10,000 images for testing.
Each image is a 28x28 grayscale image which is flattened into an array of 784 pixels. All the images belong
to one of the 10 classes as listed below:

• 0 T-shirt/top

• 1 Trouser

• 2 Pullover

• 3 Dress

• 4 Coat

• 5 Sandal

• 6 Shirt

• 7 Sneaker

• 8 Bag

• 9 Ankle boot

Instructions to run the code
To run the code, run main.py from the command line. The different architectures implemented can be
chosen by giving argument in the command line. For example,

$ python main.py architecture_1

The above command runs the code with architecture_1 defined in model.py file. The four architecture
that can be chosen are:

• architecture_1

• architecture_1_with_regularization

• architecture_2

• architecture_2_with_regularization
