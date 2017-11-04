
class NeuralNetwork(object):
    def __init__(self,i=2,h=3,o=1 ):
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

    # Forward Propogation :

'''
    def forward(self, X):
        # Propagate inputs through network
        # Matrix(z2) = Matrix(X) * Matrix(W1)

        self.z2 = np.dot(X, self.W1)

        # Applying Activation Function to each
        # element of Matrix(z2) and storing answer
        # inside a2

        self.a2 = self.activation_function(self.z2)

        # Matrix(z3) =  Matrix(a2) * Matrix(W2)

        self.z3 = np.dot(self.a2, self.W2)

        # yHat is the predicted Output Matrix
        # for given input Matrix(X)


        yHat = self.activation_function(self.z3)

        return yHat
'''

nn = NeuralNetwork()
print(nn.hiddenLayerSize)

x = np.random.randn(4,3)
for i in range(len(x)):
    print("Col : i = ",i," : ",x[i]," : ")

# plt.plot(x)
# plt.ylabel('distribution')
# plt.show()

# row wise
# zz = np.sum(x,axis=1)
# print(zz)
# col wise
# zz = np.sum(x,axis=0)
# print(zz)
# zz = np.dot(zz,(1/len(zz)))
#
# print(zz)
#
# print(sum(zz))
