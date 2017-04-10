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


win.getMouse()
win.close()
