import numpy as np
# import matplotlib.pyplot as plt


class NeuralNetwork(object):
    def __init__(self, i=2, h=3, o=1):

        # Define Parameters

        '''
        Input Layer size = Number of input
        variables

        In Our case, Number of input
        variables are :

        1) Number of hours you study
        2) Number of hours you sleep
        input Layer size must be 2

        '''

        self.inputLayerSize = i

        '''
        Hidden Layer Size I defined as 3,
        means inside the hidden layer
        there are 3 neurons 

        '''

        self.hiddenLayerSize = h

        '''
        Output Layer Size is 1 means
        there is only one output

        '''

        self.outputLayerSize = o

        '''
        Weight matrix distribution is done by
        Normal Distribution which means distribution
        is bell shaped curve with mean value = 0
        and variance = 1 which means sum of row can
        vary between(-1,1) and implimented using numpy
        np.random.randn method so that we can converge
        a result to some particular value easily for
        the given input which need to pridict
        First Parameter :- rows in matrix
        Second Parameter :- columns in matrix

        '''

        # Weight

        '''
        W1 = Weight Matrix 1 which connects input Layer to the hidden Layer
        '''

        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)

        '''
        W2 = Weight Matrix 2 which connects hidden Layer to the output Layer
        '''

        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    # Activation function helps us to represent non linear functions in order to represent 
    # complex phenomenon using neural nets otherwise it will be limited to the linear systems
        
    def activation_function(self, z):
        # Applying sigmoid activation function
        return (1 / (1 + np.exp(-z)))

    def activation_function_derrivative(self, z):
        # Gradient of sigmoid activation function
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    # Forward Propogation :

    def forward(self, X):
        # Propagate "inputs(X)" to "hiddenLayer(z2)" through network
        # using synapses "W1"
        # Matrix(z2) = Matrix(X) * Matrix(W1)

        self.z2 = np.dot(X, self.W1)

        # Applying Activation Function to each neuron
        # (element) of "hiddenLayer(z2)" and storing answer
        # inside a2

        self.a2 = self.activation_function(self.z2)

        # Matrix(z3) =  Matrix(a2) * Matrix(W2)

        self.z3 = np.dot(self.a2, self.W2)

        # yHat is the predicted Output Matrix
        # for given input Matrix(X) using 
        # set of random weighted Synapse Matrices (W1,W2)


        yHat = self.activation_function(self.z3)

        return yHat
