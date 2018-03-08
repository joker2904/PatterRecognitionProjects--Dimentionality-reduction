import numpy as np
import matplotlib.pyplot as mp
import numpy_indexed as npi
from operator import itemgetter
import time as tm

colors = ['red','green','blue','cyan','orange']

#Function to get the euclidean distance between 2 points
def EuclideanDistance(X,Y):
    return (( (X[:,0]-Y[:,0])**2 + (X[:,1]-Y[:,1])**2)**0.5 )

def LoadData(Filename):
    data = np.loadtxt(open(Filename, "rb"),delimiter=',')
    data = data.transpose()
    return data

def ShowClusters(Info,data,Centroids=None):
    mp.title(Info)
    mp.xlim(xmin=np.min(data[:, 0]) - 1, xmax=np.max(1 + data[:, 0]))
    mp.ylim(ymin=np.min(data[:, 1]) - 1, ymax=np.max(1 + data[:, 1]))
    mp.scatter(data[:,0], data[:,1], c=[colors[i] for i in (data[:,2]).astype(int)], s=5.5,marker='^',label='Points')
    if Centroids is not None:
       mp.scatter(Centroids[:,0], Centroids[:,1], c=[colors[i] for i in (Centroids[:,2]).astype(int)] , s=40,marker='*',label='Centroids')
    mp.legend()
    mp.show()

######################### Implementation of Lloyd's Algorithm #######################################
def Lloyd(k,data):
    t = 0
    done = False
    tmax = 1000
    epsilon = 0.0001
    #meu_t is the set of centroids of meu1,meu2,meu3,....meuk
    start = tm.clock()
    meu_t = np.reshape(data[np.random.choice( range(0,data.shape[0]),k,replace=False),:],(k,2))
    Ct = np.hstack((data,np.reshape(np.argsort(np.linalg.norm(data[:, None, :] - meu_t[None, :, :], axis=2), axis=1)[:, 0],(data.shape[0], 1))))
    while not done:
          #Ct is the previous Cluster set
          # meu_tplus1 is the set of updated centroids
          meu_tplus1 = (npi.group_by(Ct[:, 2]).mean(Ct))[1][:,0:2]
          # Ct is the updated cluster set
          Ctnew = np.hstack((data, np.reshape(np.argsort(np.linalg.norm(data[:, None, :] - meu_tplus1[None, :, :], axis=2), axis=1)[:, 0],(data.shape[0], 1))))
          #Tests for convergence
          #Test 1. Check if the previous and current cluster sets intersect
          Ct = np.array(sorted(Ct,key=itemgetter(2)))
          Ctnew = np.array(sorted(Ctnew,key=itemgetter(2)))
          if np.array_equal(Ct,Ctnew) == True:
             done = True
          #Test 2. Check if distance between previous and current centroid(s) is less than epsilon
          if np.reshape(EuclideanDistance(meu_t,meu_tplus1),(k,1)).all() <= epsilon:
             done = True
          #Test 3. Number of iterations exceed maximum limit
          t = t+1
          if t > tmax:
             done = True
          Ct = Ctnew
          meu_t = meu_tplus1
          end =  tm.clock()
    return Ct,np.hstack((meu_t,np.reshape( np.array(list(range(k))),(k,1)))),t,end-start
############################### End #############################################################################

######################################## Implementation of MacQueens ############################################
def MacQueens(k,data):
    start = tm.clock()
    # meu_t is teh Centroid matrix. The first 2 columns of meu_t is the 2 dimentions of the data points. The 3rd column is the ni
    meu_t = np.hstack((np.reshape(data[np.random.choice(range(0, data.shape[0]), k, replace=False), :], (k, 2)), np.reshape(np.zeros(k),(k,1))))
    for j in range(0,data.shape[0]):
        xj = np.reshape(data[j,0:2],(1,2))
        w = np.argmin(EuclideanDistance(np.reshape(meu_t[:,0:2],(k,2)),xj))
        meu_t[w,2] += 1
        meu_t[w,0:2] = meu_t[w,0:2] + ( 1/meu_t[w,2] )* (xj-meu_t[w,0:2])
    meu = np.reshape(meu_t[:,0:2],(k,2))
    Ct = np.hstack((data, np.reshape(np.argsort(np.linalg.norm(data[:, None, :] - meu[None, :, :], axis=2), axis=1)[:, 0],(data.shape[0], 1))))
    end = tm.clock()
    return Ct,np.hstack((meu,np.reshape( np.array(list(range(k))),(k,1)))),end-start
############################### End #############################################################################

######################################## Implementation of Hartigans ############################################
def Hartigans(k,data):
    start = tm.clock()
    Ct = np.hstack((data, np.reshape( np.random.choice(range(0,k), data.shape[0] ,replace=True),(data.shape[0],1))))
    meu = (npi.group_by(Ct[:, 2]).mean(Ct))[1][:,0:2]
    Converged = False
    while Converged is False:
          Converged = True
          for j in range(0,Ct.shape[0]):
              Cj = Ct[j,2]
              dmin = []
              for i in range(0,k):
                  Ct[j,2] = i
                  G = (npi.group_by(Ct[:, 2])).split(Ct)
                  dist = 0
                  #print(G)
                  for p in range(0,k):
                      t = (G[p][:, 0:2])
                      mi = np.reshape(np.mean(t, axis=0), (1,2))
                      t = np.sum((t - mi)**2,axis=1)
                      dist = dist + np.sum(t,axis=0)
                  dmin.append(dist)
              Cw = np.argmin(dmin)
              if Cw != Cj:
                 Converged = False
                 Ct[j,2] = Cw
                 meu = (npi.group_by(Ct[:, 2]).mean(Ct))[1][:, 0:2]
              else:
                 Ct[j,2] = Cj
    end = tm.clock()
    return Ct, np.hstack((meu, np.reshape(np.array(list(range(k))), (k, 1)))), end - start
########################## End #################################################################################

################################### Implementation of Spectral Clusteral Clustering ############################
def Spectral(beta,data):
    start = tm.clock()
    Sij = np.exp( (-1.0 * beta) * (np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2))**2 )
    Dij = np.reshape(np.sum(Sij,axis = 1),(Sij.shape[0],1)) * np.identity(Sij.shape[0])
    L = Dij-Sij
    w,u = np.linalg.eigh(L)
    FiedlerVector = ( u[:,np.argsort(w)[3]] )
    FiedlerVector[ FiedlerVector < 0 ] = 0
    FiedlerVector[ FiedlerVector > 0 ] = 1
    Clusters = np.hstack((data,np.reshape( FiedlerVector, (FiedlerVector.shape[0],1))))
    end = tm.clock()
    return Clusters,end-start
################################## End ########################################################################

Points1 = LoadData('data-clustering-1.csv')
Points2 = LoadData('data-clustering-2.csv')


for i in range(1,7):
    ClusterPoints,time = Spectral(i,Points2)
    ShowClusters('Clusters For Non Convex Data, for Beta = '+str(i)+' Time = '+str(time),ClusterPoints)

avg_time = 0
k = 3
for i in range(0,10):
    ClusterPoints,meu,NoOfIterations,time = Lloyd(k,Points1)
    #print(ClusterPoints,meu,t)
    avg_time = avg_time + time
    ShowClusters('Clusters For Llyods Algorithm, k='+str(k)+' Means. Time = '+str(time),ClusterPoints,meu)
print("\n Avergage Time of Lyod's Algorithm =",avg_time/10)

avg_time = 0
k = 2
for i in range(0,3):
    ClusterPoints,meu,NoOfIterations,time = Lloyd(k,Points2)
    #print(ClusterPoints,meu,Points.shape)
    avg_time = avg_time + time
    ShowClusters('Clusters For Llyods Algorithm, k='+str(k)+' Means. Time = '+str(time),ClusterPoints,meu)
print("\n Avergage Time =",avg_time/3)

avg_time = 0
k = 3
for i in range(0,10):
    ClusterPoints,meu,time = MacQueens(k,Points1)
    #print(ClusterPoints,meu,Points.shape)
    avg_time = avg_time + time
    ShowClusters('Clusters For MacQueens Algorithm, k='+str(k)+' Means. Time = '+str(time),ClusterPoints,meu)
print("\n Avergage Time for MacQueen's Algorithm =",avg_time/10)

avg_time = 0
k = 2
for i in range(0,3):
    ClusterPoints,meu,time = MacQueens(k,Points2)
    #print(ClusterPoints,meu,Points.shape)
    avg_time = avg_time + time
    ShowClusters('Clusters For MacQueens Algorithm, k='+str(k)+' Means. Time = '+str(time),ClusterPoints,meu)
print("\n Avergage Time =",avg_time/3)

avg_time = 0
k = 3
for i in range(0,10):
    ClusterPoints,meu,time = Hartigans(k,Points1)
    #print(ClusterPoints,meu,t)
    avg_time = avg_time + time
    ShowClusters('Clusters For Hartigans Algorithm, k='+str(k)+' Means. Time = '+str(time),ClusterPoints,meu)
print("\n Avergage Time for Hartigan's Algorithm =",avg_time/10)

avg_time = 0
k = 2
for i in range(0,3):
    ClusterPoints,meu,time = Hartigans(k,Points2)
    #print(ClusterPoints,meu,Points.shape)
    avg_time = avg_time + time
    ShowClusters('Clusters For Hartigans Algorithm, k='+str(k)+' Means. Time = '+str(time),ClusterPoints,meu)
print("\n Avergage Time =",avg_time/3)
'''
####### time comparision ###########
kL = [3,4,5,6,7,8,9,10]
t1 =[]
t2=[]
t3=[]
for k in range(3,11):
    ClusterPoints, meu, NoOfIterations, time1 = Lloyd(k, Points1)
    ClusterPoints, meu, time2 = Hartigans(k, Points1)
    ClusterPoints, meu, time3 = MacQueens(k, Points2)
    t1.append(time1)
    t2.append(time2)
    t3.append(time3)

mp.title("abcd")
mp.plot(kL,time1)
mp.plot(kL,time2)
mp.plot(kL,time3)
mp.show()
'''