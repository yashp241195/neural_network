import numpy as np





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

'''

Normalization by scaling our Data :

X = X/max(X)

Y = Y/max(Y), where max(Y) is given as 100

'''


X /= np.amax(X, axis=0)
Y /= 100

print '\nInput Matrix \n X = \n', X, ' \n\n \n Output Matrix\n Y = \n', Y



class NeuralNetwork(object):
    def __init__(self):
        # Define HyperParameters

        '''

        Input Layer size = Number of input
        variables
        
        In Our case, Number of input
        variables are :
        
        1) Number of hours you study
        2) Number of hours you sleep

        input Layer size must be 2
        
        '''

        self.inputLayerSize = 2
  
        '''

        Hidden Layer Size I defined as 3,
        means inside the hidden layer
        there are 3 neurons 
        
        '''

        self.hiddenLayerSize = 3

        '''

        Output Layer Size is 1 means
        there is only one output
        
        '''

        self.outputLayerSize = 1

        
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

    # Forward Propogation :

    
    def forward(self, X):
        
        # Propagate inputs through network
        # Matrix(z2) = Matrix(X) * Matrix(W1)

        self.z2 = np.dot(X, self.W1)

        # Applying Activation Function to each
        # element of Matrix(z2) and storing answer
        # inside a2
        
        self.a2 = self.activation_function(self.z2)

        # Matrix(z3) =  Matrix(a2) * Matrix(W2)

        self.z3 = np.dot(self.a2 , self.W2)

        # yHat is the predicted Output Matrix
        # for given input Matrix(X) 
        

        yHat = self.activation_function(self.z3)

        
        return yHat

    
    # Computing the cost Function

    def costFunction(self, X, Y):

        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        
        J = 0.5*sum((Y-self.yHat)**2)

        return J

    # Computing dJ/dW1 and dJ/dW2 for minimization

    def costFunctionPrime(self, X, y):
        
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.activation_function_derrivative(self.z3))
        # for matrix M if I write M.T it means Transpose
        
        dJdW2 = np.dot(self.a2.T, delta3)
        
        
        delta2 = np.dot(delta3, self.W2.T) * self.activation_function_derrivative(self.z2)
        
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2


    # Activation Function 
    
    def activation_function(self, z):
        
        # Applying sigmoid activation function
        return (1 / (1 + np.exp(-z)))

    def activation_function_derrivative(self,z):
        
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    

NN = NeuralNetwork()
yHat = NN.forward(X)

print '\n Forward Propogation Output Prediction Matrix(yHat) : \n',yHat
print '\n\n Cost Matrix :\n 1/2 (Matrix(Y) - Matrix(yHat))**2 \n\n ',NN.costFunction(X,Y)
print '\n\n Minimizing Cost Function by Gradient Descent '
