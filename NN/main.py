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
    
    '''
    Normalization by scaling our Data :
    X = X/max(X)
    Y = Y/max(Y), where max(Y) is given as 100
    '''
    
    
    X /= np.amax(X, axis=0)
    Y /= 100
    
    #print('\nInput Matrix \n X = \n', X, ' \n\n \n Output Matrix\n Y = \n', Y)
    
    
if __name__== "__main__":
  main()
