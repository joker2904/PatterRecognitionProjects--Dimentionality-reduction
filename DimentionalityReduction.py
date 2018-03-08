import numpy as np
import matplotlib.pyplot as mp
import numpy_indexed as npi
from operator import itemgetter
import time as tm
from mpl_toolkits import mplot3d
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

colors = ['cyan','green','blue','red','orange']

def LoadData(Filename):
    data = np.loadtxt(open(Filename, "rb"),delimiter=',')
    data = data.transpose()
    return data

def Show2D(Info,data):
    mp.title(Info)
    mp.xlim(xmin=np.min(data[:, 0]) - 1, xmax=np.max(1 + data[:, 0]))
    mp.ylim(ymin=np.min(data[:, 1]) - 1, ymax=np.max(1 + data[:, 1]))
    mp.scatter(data[:,0], data[:,1], c=[colors[i] for i in (data[:,2]).astype(int)], s=8.5)
    mp.legend()
    mp.show()

def Show3D(Info,data):
    fig = mp.figure()
    ax = mplot3d.Axes3D(fig)
    ax.set_title(Info)
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c=[colors[i] for i in (data[:,3]).astype(int)], cmap='Greens')
    mp.show()

########## Implementation of PCA Algorithm ####################################
def PCA(dimension,data):
    X = data - np.reshape(np.mean(data,axis=1),(data.shape[0],1))
    C = (1/data.shape[0]) * np.dot(X,X.transpose())
    eigenValues, eigenVectors = np.linalg.eigh(C)
    U = ( eigenVectors[:, np.argsort(-eigenValues)[0:dimension] ] )
    return np.dot(U.transpose(),X).transpose()
######### End of PCA Algorithm ################################################

######## Implementatipon of LDA Algorithm ####################################
def LDA(dimension,data):
    # Compute mean of the whole dataset
    m = np.reshape(np.mean(data[:,0:data.shape[1]-1],axis=0),(data.shape[1]-1,1))
    #    m = m[0:m.shape[0],:]
    # Compute Sw, the covariant of classified means
    # Compute Sb
    G = (npi.group_by(data[:,data.shape[1]-1])).split(data)
    Sw = np.reshape(np.zeros((data.shape[1]-1 )*(data.shape[1]-1 )),(data.shape[1]-1 ,data.shape[1]-1 ))
    Sb = np.reshape(np.zeros((data.shape[1]-1 )*(data.shape[1]-1 )),(data.shape[1]-1 ,data.shape[1]-1 ))
    for i in range(0,G.shape[0]):
        t = (G[i][:,0:G[i].shape[1]-1])
        mi = np.reshape(np.mean(t,axis=0),(1,t.shape[1]))
        t = (t - mi)
        Sw = Sw + np.dot(t.transpose(),t)
        Sb = Sb + (t.shape[0])*np.dot((mi-m).transpose(),(mi-m))
    Projection = np.dot(np.linalg.pinv(Sw),Sb)
    #Compute the eigen values and eigen vectors , which will be used as the projection matrix
    eigenValues, eigenVectors = np.linalg.eigh( Projection.transpose() )
    U = (eigenVectors[:, np.argsort(-eigenValues)[0:dimension]])
    X = data[:,0:data.shape[1]-1].transpose()
    return np.dot(U.transpose(), X).transpose()
######### End of LDA Algorithm ################################################

######## Implementatipon of LDA Algorithm ( changed parameters, used for experimentation ) ####################################
def LDA2(dimension,data):
    m = np.reshape(np.mean(data[:,0:data.shape[1]-1],axis=0),(data.shape[1]-1,1))
    G = (npi.group_by(data[:,data.shape[1]-1])).split(data)
    Sw = np.reshape(np.zeros((data.shape[1]-1 )*(data.shape[1]-1 )),(data.shape[1]-1 ,data.shape[1]-1 ))
    Sb = np.reshape(np.zeros((data.shape[1]-1 )*(data.shape[1]-1 )),(data.shape[1]-1 ,data.shape[1]-1 ))
    for i in range(0,G.shape[0]):
        t = (G[i][:,0:G[i].shape[1]-1])
        mi = np.reshape(np.mean(t,axis=0),(1,t.shape[1]))
        t = (t - mi)
        Sw = Sw + np.dot(t.transpose(),t)
        Sb = Sb + (t.shape[0])*np.dot((mi-m).transpose(),(mi-m))
    Projection = np.dot(np.linalg.inv(Sw),Sb)
    #Compute the eigen values and eigen vectors , which will be used as the projection matrix
    eigenValues, eigenVectors = np.linalg.eigh( Projection.transpose() )
    print(eigenValues[np.argsort(-eigenValues)[0] ])
    U = (eigenVectors[:,np.argsort(-eigenValues)[0:dimension]])
    X = data[:,0:data.shape[1]-1]
    return np.dot(U.transpose(), (X-m.transpose()).transpose()).transpose()
######### End of LDA Algorithm ################################################

#########LDA using Sklearn........for verification############
def LDA_sklearn(data,classification):
    lda = LinearDiscriminantAnalysis(n_components=2)
    X = lda.fit(data, classification).transform(data)
    return X
##############################################################

DPoints = LoadData('data-dimred-X.csv')
Classes = LoadData('data-dimred-y.csv')
data = np.hstack( (DPoints, np.reshape(Classes,(DPoints.shape[0],1) )))
#print(DPoints.shape,Classes.shape)

########## Implement PCA for 2 and 3 dimensions ##################################################
RD2 = PCA(2,DPoints.transpose())
Plotter = np.hstack( (RD2, np.reshape(Classes,(RD2.shape[0],1) )))
Show2D("PCA - 2D Projections",Plotter)

RD3 = PCA(3,DPoints.transpose())
Plotter = np.hstack( (RD3, np.reshape(Classes,(RD3.shape[0],1) )))
Show3D("PCA - 3D Projections",Plotter)

######### Implement Fischers Linear Discriminant algorithm , for 2 and 3 dimensions###############
LD2 = LDA(2,data)
Plotter = np.hstack( (LD2, np.reshape(Classes,(LD2.shape[0],1) )))
Show2D("LDA - 2D Projections(uning pinv)",Plotter)

LD3 = LDA(3,data)
Plotter = np.hstack( (LD3, np.reshape(Classes,(LD3.shape[0],1) )))
Show3D("LDA - 3D Projections(using pinv)",Plotter)

LD2 = LDA2(2,data)
Plotter = np.hstack( (LD2, np.reshape(Classes,(LD2.shape[0],1) )))
Show2D("LDA - 2D Projections(uning inv)",Plotter)

LD3 = LDA2(3,data)
Plotter = np.hstack( (LD3, np.reshape(Classes,(LD3.shape[0],1) )))
Show3D("LDA - 3D Projections(using inv)",Plotter)

######## Verify the LDA for 2-D projections using SKLearn#################
LDA_sk2 = LDA_sklearn(DPoints,Classes)
Plotter = np.hstack( (LDA_sk2, np.reshape(Classes,(LDA_sk2.shape[0],1) )))
Show2D("LDA SKLEARN- 2D Projections",Plotter)
print(LDA_sk2)

