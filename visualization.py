from graphics import *
win = GraphWin('G',1200,650)

inputLayerCode = 0
outputLayerCode = 1
hiddenLayerCode = 2

text = "Predicting Test Scores using Artificial Neural Networks \n with Supervised Regression Technique "

pt = Point(550,30)
label = Text(pt, text)
label.setSize(15)
label.setStyle("bold")
label.draw(win)

text = "Input Layer "

pt = Point(70,100)
label = Text(pt, text)
label.setSize(15)
label.setStyle("bold")
label.draw(win)

text = "Hidden Layer "

pt = Point(250,100)
label = Text(pt, text)
label.setSize(15)
label.setStyle("bold")
label.draw(win)

text = "Output Layer "

pt = Point(470,100)
label = Text(pt, text)
label.setSize(15)
label.setStyle("bold")
label.draw(win)


text = "Marks Scored\n(out of 100)"

pt = Point(470,200)
label = Text(pt, text)
label.setSize(10)
label.setStyle("bold")
label.draw(win)

text = " \n 1) Number of Hours\n you Study \n 2) Number of Hours\n you Sleep "

pt = Point(70,160)
label = Text(pt, text)
label.setSize(10)
label.setStyle("bold")
label.draw(win)



class Neuron(object):
    def __init__(self,x,y,text,LayerType):
          
        self.radius = len(text) * 5
        self.x = x
        self.y = y
                
        pt = Point(self.x,self.y)

        cir = Circle(pt, self.radius)
        label = Text(pt, text)

        if LayerType is 0:
            cir.setFill('green')
            label.setFill('blue')
            
        elif LayerType is 1:
            cir.setFill('red')
            label.setFill('white')
        else :
            cir.setFill('black')
            label.setFill('white')

        cir.draw(win)
  
        
        label.setSize(10)
        label.setStyle("bold")

        label.draw(win)






class drawLayer(object):
    def __init__(self,x,y,Arr,LayerType):
        
        self.coordinateX = []
        self.coordinateY = []
        
        for i in range(0,len(Arr)):

            self.coordinateX.append(x)
            self.coordinateY.append(y)
            
            Neuron(x,y,Arr[i],LayerType)
            y+=100

    
def JoinLayer(X1,Y1,X2,Y2):
    limX1 = len(X1)
    limX2 = len(X2)

    for i in range(0,limX1):
        for j in range(0,limX2):
            pt1 = Point(X1[i]+35,Y1[i])
            pt2 = Point(X2[j]-35,Y2[j])

            line = Line(pt1,pt2)
            line.draw(win)
    


Input_Arr = [' Study ',' Sleep ']
Output_Arr = [' Marks  ']
Hidden_Arr = ['Hidden1','Hidden2','Hidden3']
        
InputLayer = drawLayer(50,250,Input_Arr,inputLayerCode)

X1 = InputLayer.coordinateX
Y1 = InputLayer.coordinateY

HiddenLayer = drawLayer(250,200,Hidden_Arr,hiddenLayerCode)

X2 = HiddenLayer.coordinateX
Y2 = HiddenLayer.coordinateY

OutputLayer = drawLayer(450,300,Output_Arr,outputLayerCode)

X3 = OutputLayer.coordinateX
Y3 = OutputLayer.coordinateY

JoinLayer(X1,Y1,X2,Y2)
JoinLayer(X2,Y2,X3,Y3)





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



text = "Input data\n\t\t \t\t \t\t X(Study(Hours))\t X(Sleep(Hours)) \t\t Y(Marks)"

pt = Point(670,100)
label = Text(pt, text)
label.setSize(10)
label.setStyle("bold")
label.draw(win)


xx = []
xx.append(np.amax(X, axis=0))

text = "\n\n Normalized data \nAfter Scaling \n\n\n \t\t\t\t \t\t X(Study(Hours))\t X(Sleep(Hours)) \t\t Y(Marks)\n \t\t\t\t X=X/"+str(xx[0][0])+" \t X=X/"+str(xx[0][1])


pt = Point(670,300)
label = Text(pt, text)
label.setSize(10)
label.setStyle("bold")
label.draw(win)


text = "Y=Y/100"

pt = Point(980,355)
label = Text(pt, text)
label.setSize(10)
label.setStyle("bold")
label.draw(win)



def printData(X,Y,YPos):
    for i in range(0,len(X)):
        for j in range(0,2):
            
            pt = Point(750+100*j,YPos+i*50)
            text = str(X[i][j])
            
            text += "\t\t"
            label = Text(pt, text)
            label.setSize(10)
            label.setStyle("bold")
            label.draw(win)

    for i in range(0,len(Y)):
        for j in range(0,1):
            
            pt = Point(1000+100*j,YPos+i*50)
            text = str(Y[i][j])
            
            text += "\t\t"
            label = Text(pt, text)
            label.setSize(10)
            label.setStyle("bold")
            label.draw(win)

        
    

printData(X,Y,150)

'''

Normalization by scaling our Data :

X = X/max(X)

Y = Y/max(Y), where max(Y) is given as 100

'''


X /= np.amax(X, axis=0)
Y /= 100

printData(X,Y,400)


win.getMouse()
win.close()
