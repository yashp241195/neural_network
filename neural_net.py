import numpy as np
import time


X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
Y = np.array(([75], [82], [93]), dtype=float)

X /= np.amax(X, axis=0)
Y /= 100

#print ('X = ', X, ' \n Y = ', Y)


class NeuralNetwork(object):
    def __init__(self):
        # Define HyperParameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # weight
        self.W1 = np.random.rand(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.rand(self.hiddenLayerSize,self.outputLayerSize)

    def forward(self, X):
        # Propagate inputs through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2 , self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        # Applying sigmoid activation function
        return 1 / (1 + np.exp(-z))

NN = NeuralNetwork()
yHat = NN.forward(X)
#print (yHat)

# Total Weights in neural net

TotalWeights = (NN.inputLayerSize*NN.hiddenLayerSize) + (NN.hiddenLayerSize*NN.outputLayerSize)
MaxItr = 10 * TotalWeights


#Uniform distribution of 1000 numbers between -10 to +10
weightsToTry = np.linspace(-10,10,MaxItr)

#Cost matrix with 100 zeros
cost = np.zeros(MaxItr)

#Applying Brute force Optimization

#Optimize for one weight out of Total Weights W[0][0]

for i in range(0,MaxItr):
    NN.W1[0][0] = weightsToTry[i];
    yHat = NN.forward(X)
    cost[i]=0.5*sum((Y-yHat)**2)

print Y-yHat

# for nine weights we need 90^9 iterations

# Due to high time Complexity this will not be the Solution

# Optimizing Cost function
# by using Partial derrivatives to predict the direction of searching Value of weight

# d/dW2(cost) = d/dW2(0.5*sum((Y-yHat)**2))

# we need sigmoid derrivative also
# these must be added to class Neural Network 

def sigmoidPrime(self, z):
    # Applying sigmoid activation function
    return np.exp(-z) / ((1 + np.exp(-z))**2)

#derrivative of cost function
def costFunctionPrime(self, X, Y):
    #Compute derrivative wrt W1 and W2
    self.yHat=self.forward(X)

    delta3 = np.multiply(-(Y-self.yHat), self.sigmoidPrime(self.z3))
    dJdW2 = np.dot(self.a2.T, delta3)

    delta2 = np.multiply(delta3, self.W2.T)*self.sigmoidPrime(self.z3)
    dJdW1 = np.dot(X.T, delta3)




