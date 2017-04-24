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

        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)

        # Normalizing the cost Function
        
        J = 0.5*sum((Y-self.yHat)**2)

        return J

    # Backpropogation of errors 

    # Computing partial derrivatives dJ/dW1 and dJ/dW2 for minimization

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


    # Gradient Descent

    def computeGradients(self, X, y):
        
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)

        # ravel is used to convert 2D to 1D array
        # In this we are using row wise Left to Right
        # evaluation in order to 
        # returning the concatinated single array

        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


    # Roll W1 and W2 into single column

    def getParams(self):
        # Get W1 and W2 unrolled into vector:
        # ravel is used to convert 2D to 1D array
        # In this we are using row wise Left to Right
        
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))

        return params

    # Rollback W1 and W2 into it's initial conditions
    # means W1 , W2 will be no more a 1D array it will
    # reforms it's rows,col counts 2D array
    
    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.

        # reshaping the W1 and W2 in their initial conditions
        W1_start = 0

        W1_end = self.hiddenLayerSize * self.inputLayerSize

        # reshaping W1
        
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize

        # reshaping W2
        
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

#

    def computeNumericalGradient(N, X, y):
        
            paramsInitial = N.getParams()
            
            numgrad = np.zeros(paramsInitial.shape)
            
            perturb = np.zeros(paramsInitial.shape)
            
            e = 1e-ArithmeticError

            for p in range(len(paramsInitial)):
                #Set perturbation vector
                
                perturb[p] = e

                N.setParams(paramsInitial + perturb)
                loss2 = N.costFunction(X, y)
                
                N.setParams(paramsInitial - perturb)
                loss1 = N.costFunction(X, y)

                # Compute Numerical Gradient
                # Cauchy Limits formula
                # f'(x)=(f(x+e)-f(x-e))/2e ,where e->0
                
                numgrad[p] = (loss2 - loss1) / (2*e)

                #Return the value we changed to zero:
                perturb[p] = 0
                
            #Return Params to original value:
            N.setParams(paramsInitial)

            return numgrad















## ----------------------- Part 6 ---------------------------- ##
from scipy import optimize


class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

    

    

NN = NeuralNetwork()
yHat = NN.forward(X)

print '\n Forward Propogation Output Prediction Matrix(yHat) : \n',yHat
print '\n\n Cost Matrix :\n 1/2 (Matrix(Y) - Matrix(yHat))**2 \n\n ',NN.costFunction(X,Y)
print '\n\n Minimizing Cost Function by Gradient Descent '
