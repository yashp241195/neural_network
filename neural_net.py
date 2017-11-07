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

    def activation_function(self, z):
        # Applying sigmoid activation function
        return (1 / (1 + np.exp(-z)))

    def activation_function_derrivative(self, z):
        # Gradient of sigmoid activation function
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    # Forward Propogation :

    def forward(self, X):
        '''
        Propagate "inputs(X)" to "hiddenLayer(z2)" through network
        using synapses "W1"
        Matrix(z2) = Matrix(X) * Matrix(W1)
        '''
        self.z2 = np.dot(X, self.W1)

        '''
        Applying Activation Function to each neuron
        (element) of "hiddenLayer(z2)" and storing answer
        inside a2
        '''

        self.a2 = self.activation_function(self.z2)

        # Matrix(z3) =  Matrix(a2) * Matrix(W2)

        self.z3 = np.dot(self.a2, self.W2)

        # yHat is the predicted Output Matrix
        # for given input Matrix(X) using random data


        yHat = self.activation_function(self.z3)

        return yHat

    # Computing the cost Function

    def costFunction(self, X, Y):
        '''
        Cost function is taken as ((y-yh)**2)/2
        because quadratic equations are convex in
        nature in which either roots are equal
        or pair of (maxima/minima)
        '''
        
        self.yHat = self.forward(X)

        J = 0.5 * ((Y - self.yHat) ** 2)
        # total error
        total_error = np.sum(J)
        return total_error

    # Backpropogation of errors

    def costFunctionPrime(self, X, Y):

        self.yHat = self.forward(X)
    
        # Computing errors
        delta3 = np.multiply(-(Y-self.yHat), self.activation_function_derrivative(self.z3))
        # for matrix M if I write M.T it means Transpose
        delta2 = np.dot(delta3, self.W2.T) * self.activation_function_derrivative(self.z2)
        
        dJdW2 = np.dot(self.a2.T, delta3)               
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2

    def BackPropogation(self, X, Y, scalar = 3):

        cost1 = self.costFunction(X,Y)

        dJdW1, dJdW2 = self.costFunctionPrime(X,Y)

        self.W1 -= (scalar*dJdW1)
        self.W2 -= (scalar*dJdW2)

        cost2 = self.costFunction(X,Y)

        return cost1,cost2

    def train(self,X,Y,iteration = 60000):
        print("Initial Cost before training : ",self.BackPropogation(X, Y))
        for i in range(iteration):
            self.BackPropogation(X, Y)
        print("Final Cost after training : ", self.BackPropogation(X, Y))



def main():

    # Artificial Neural Network :

    '''
    Suppose you want to predict your test score
    from input given as :
        X = [Number of hours you study , Number of hours you sleep]
        Y = [Test on score out of 100]
    X - Input :
    -> X = [Number of hours you study , Number of hours you sleep]
    Y - Output :

    -> Y = [Test Score]
    '''

    X = np.array(([3, 5], [5, 1], [10, 2]), dtype = float)

    Y = np.array(([75], [82], [93]), dtype = float)
    print("Artificial Neural Network Implementation from Scratch")
    print("\nInput (study,sleep) hours \n", X)
    print("\nOutput (Score( out of 100)) \n", Y)

    '''
    Normalization by scaling our Data :
    X = X/max(X)
    Y = Y/max(Y), where max(Y) is given as 100
    '''


    X /= np.amax(X, axis=0)
    Y /= 100



    NN = NeuralNetwork()
    yHat = NN.forward(X)

    print('\n Forward Propogation Output Prediction Matrix(yHat) : \n',yHat)
    J = NN.costFunction(X,Y)
    print('\n Cost Function value : \n', J)

    NN.train(X,Y)

    print('\nAfter Training Output Prediction Matrix(yHat) : \n', 100*NN.yHat)

if __name__== "__main__":
  main()
