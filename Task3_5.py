import numpy as np
import numpy.linalg as la
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

#Load the data
def LoadData():
    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)
    OriginalData = data[:, 0:2].astype(np.float)
    # read gender data into 1D array (i.e. into a vector)
    Gender = data[:, 2]

    # removing negative and zeros from both columns
    X = OriginalData[OriginalData[:, 1] > 0, :]
    X = X[X[:, 0] > 0, :]

    # Get the list of outliers
    Outliers = OriginalData[OriginalData[:, 1] > 0, :]
    Outliers = Outliers[Outliers[:, 0] < 0, :]
    return X,Outliers[:,-1]

Sample,Outliers = LoadData()
hgt = Sample[:,1]
wgt = Sample[:,0]

xmin = hgt.min()-15
xmax = hgt.max()+15
ymin = wgt.min()-15
ymax = wgt.max()+15

def plot_data_and_fit(Info,h, w, x, y):
    plt.title(Info)
    plt.plot(h, w, 'ko', x, y, 'r-')
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.show()

## Scaling the values ####
def trsf(x):
    return x / 100.

n = 10
x = np.linspace(xmin, xmax, 500)

# method 1:
# regression using ployfit
c = poly.polyfit(hgt, wgt, n)
y = poly.polyval(x, c)
plot_data_and_fit('Polyfit plot - Almost Correct Fit',hgt, wgt, x, y)

# method 2:
# regression using the Vandermonde matrix and pinv
X = poly.polyvander(hgt, n)
c = np.dot(la.pinv(X), wgt)
y = np.dot(poly.polyvander(x,n), c)
plot_data_and_fit('Polyvander and pinv - Incorrect Fit ( Numerical errors in Numpy )',hgt, wgt, x, y)

# method 3:
# regression using the Vandermonde matrix and lstsq
X = poly.polyvander(hgt, n)
c = la.lstsq(X, wgt)[0]
y = np.dot(poly.polyvander(x,n), c)
plot_data_and_fit('Polyvander and Lstsqr - Incorrect Fit ( Numerical errors in Numpy) ',hgt, wgt, x, y)

# method 4:
# regression on transformed data using the Vandermonde
# matrix and either pinv or lstsq
X = poly.polyvander(trsf(hgt), n)
c = np.dot(la.pinv(X), wgt)
y = np.dot(poly.polyvander(trsf(x),n), c)
plot_data_and_fit('Polyvander and pinv, using scaled values ( almost accurate plot, similar to polyfit )',hgt, wgt, x, y)

# method 5:
# regression using the Vandermonde matrix and lstsq
X = poly.polyvander(trsf(hgt), n)
c = la.lstsq(X, wgt)[0]
y = np.dot(poly.polyvander(trsf(x),n), c)
plot_data_and_fit('Polyvander and Lstsqr, using scaled values ( almost accurate plot, similar to polyfit )',hgt, wgt, x, y)
