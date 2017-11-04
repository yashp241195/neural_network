# One Hidden Layer

class NeuralNetwork(object):
    def __init__(self,inputs=2,hiddenLayerSize=3,outputs=1 ):
        # Define HyperParameters

        '''
        Input Layer size = Number of input
        variables

        In Our case, Number of input
        variables are :

        1) Number of hours you study
        2) Number of hours you sleep
        Input Layer size must be 2

        '''

        self.inputLayerSize = inputs

        '''
        Hidden Layer Size I defined as 3,
        means inside the hidden layer
        there are 3 neurons 

        '''

        self.hiddenLayerSize = hiddenLayerSize

        '''
        Output Layer Size is 1 means
        there is only one output

        '''

        self.outputLayerSize = outputs

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


nn = NeuralNetwork()
