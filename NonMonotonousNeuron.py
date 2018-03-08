import numpy as np
import matplotlib.pyplot as mp
import numpy_indexed as npi
from operator import itemgetter
import time as tm
from sklearn import svm,datasets
from matplotlib.colors import ListedColormap

#Plot the contours and decision surface of trained SVM
def plot_SVM_Decision(Data, Classes,c,Info):
   markers = ('x', 'o', '^', 'o', 'o')
   colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
   cmap = ListedColormap(colors[:len(np.unique(Classes))])
   # plot the decision surface boundaries
   x1_min, x1_max = Data[:, 0].min() - 1, Data[:, 0].max() + 1
   x2_min, x2_max = Data[:, 1].min() - 1, Data[:, 1].max() + 1
   #Use meshgrid
   m1, m2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),np.arange(x2_min, x2_max, 0.02))
   Z = c.predict(np.array([m1.ravel(), m2.ravel()]).T)
   Z = Z.reshape(m1.shape)
   #Plot contours
   mp.contourf(m1, m2, Z, alpha=0.4, cmap=cmap)
   mp.xlim(m1.min(), m1.max())
   mp.ylim(m2.min(), m2.max())
   # plot all samples
   for i, classes in enumerate(np.unique(Classes)):
       mp.scatter( x=Data[Classes == classes, 0], y=Data[Classes == classes, 1], alpha=0.8, c=cmap(i), marker=markers[i], label=classes)
   mp.title(Info)
   mp.show()


colors = ['red','green','blue']
def LoadPoints():
    Points = np.loadtxt(open('xor-X.csv', "rb"), delimiter=',').transpose()
    Classes = np.reshape(np.loadtxt(open('xor-y.csv', "rb"), delimiter=','),(Points.shape[0],1))
    return np.hstack((Points,Classes)),Points.shape[0]

#Plot the decision hyperplane of the learning perceptron
def PartitionLine(x,a,b,Theta,position):
    Root = ((-2)*np.log(1/2))**0.5
    return (-a/b)*x - (position)*(Root/b) - (Theta/b)

#Show the trained points of perceptron learning
def Show2D(Info,data,partition=False,w1=0,w2=0,Theta=0):
    mp.title(Info)
    mp.xlim(xmin=np.min(data[:, 0]) - 1, xmax=np.max(1 + data[:, 0]))
    mp.ylim(ymin=np.min(data[:, 1]) - 1, ymax=np.max(1 + data[:, 1]))
    clist = (data[:,2] <= 0).astype(int)
    mp.scatter(data[:,0], data[:,1], c=[colors[i] for i in clist], s=8.5)
    x =  np.linspace(-3,3,500)
    if partition is True:
       mp.plot(x,PartitionLine(x,w2,w1,Theta,1.0),'b')
       mp.plot(x,PartitionLine(x,w2,w1,Theta,-1.0),'b')
    mp.legend()
    mp.show()

# A non monotonous activation function
def ActivationFunction(z):
    return (2.0 * np.exp( -0.5*z*z)) - 1.0

def WSummation(x):
    return x*np.exp(-0.5*x*x)

#Train the perceptron using gradient descent
def Train(NoOfIterations,data,N,ShowIterationProgress=False):
    #Values of N_theta and N_w are as given in the problem statement
    N_theta = 0.001
    N_w = 0.005
    #Values of the initial weight matrix w and Theta are randomly assumed
    w = np.reshape(np.array([0.5,0.45]),(2,1))
    Theta = -0.2

    x = data[:,0:2].transpose()
    f = np.vectorize(ActivationFunction)
    g = np.vectorize(WSummation)
    OutputMatrix = data[:,2]
    for i in range(0,NoOfIterations):
        NetM = np.reshape( np.dot(w.transpose(),x) - Theta,(N,1) )
        OutputMatrix = np.reshape( f(NetM),(N,1) )
        #print(np.hstack((NetM,(OutputMatrix>0).astype(int))))
        Y =  OutputMatrix - np.reshape(data[:,2],(N,1)) # y(xi) - yi
        S = np.multiply(Y,g(NetM))
        w = w + (N_w)*np.dot(x,S)
        Theta = Theta - N_theta * np.sum(S,axis=0)
        #Iteration progress
        if ShowIterationProgress is not False:
           Show2D("Training in progress ::" + str(i) + " th iteration :::",
                   np.hstack((data[:,0:2],OutputMatrix)), False, w[0, 0], w[1, 0], Theta)
           print("\n Coefficients =", w)
           print("\n Theta =", Theta)
    return np.hstack((data[:,0:2],OutputMatrix)),w,Theta

#Function to load the data points and the classes and implement the perceptron training
def PerceptronTrain():
    p,n = LoadPoints()
    Show2D("XOR Problem :: Original Points and their Classes ::", p)
    #Show the training progress of the Perceptron at each level :
    trainedData,w,Theta = Train(10,p,n,True)
    print("\n Coefficients =",w)
    print("\n Theta =",Theta)
    Show2D("Training in progress ::" + str(10) + " th iteration :::", trainedData, True, w[0, 0], w[1, 0], Theta)

def PolynomialSVMTrain(d):
    p,n = LoadPoints()
    c = svm.SVC(kernel='poly',degree=d,C=1.0)
    c.fit(p[:,0:2],p[:,2])
    plot_SVM_Decision(p[:,0:2],p[:,2],c,'SVM Training, Polynomial Kernal of Degree ='+str(d))



#Train a SVM for the XOR data, using a polynomial function of degree d ( using SKLearn )
PolynomialSVMTrain(2)
PolynomialSVMTrain(3)
PolynomialSVMTrain(4)
PolynomialSVMTrain(5)
PolynomialSVMTrain(10)


#Train a perceptron for the XOR data
PerceptronTrain()
