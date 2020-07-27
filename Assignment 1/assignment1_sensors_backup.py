# -*- coding: utf-8 -*-
"""Assignment1_Sensors.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ftF9_4nVgUtGdKd0OFuLRP90zY1X1Ok4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

"""**We 're gonna upload the file case4 using google Collab**"""

uploaded = files.upload()

print (uploaded)

import io
data = pd.read_csv(io.StringIO(uploaded['case4.csv'].decode('utf-8')))
print(data)

"""**Giving name to  the columns of the data frame:**"""

data.columns=['x', 'y']
print(data)

"""**Ploting the data frame:**"""

fig = plt.figure() 
plt.scatter(data.x, data.y, label='acceleration vs time')
plt.xlabel("x ")
plt.ylabel("y ")
plt.legend()
plt.title('dataset')

"""**Calculating the mean of x and y:**"""

mean_x, mean_y=data.mean()
print(mean_x, mean_y)

"""**Calculating the standard deviation:**"""

sd_x, sd_y=data.std()
print(sd_x, sd_y)

"""**Exclusion of the rare random deviations**"""

data= data[data.y<3*sd_y]

fig = plt.figure() 
plt.scatter(data.x, data.y, label='acceleration vs time')
plt.xlabel("x ")
plt.ylabel("y ")
plt.legend()
plt.title('dataset after removing the rare random deviations')

"""**Calculating the confidence interval:**
1. Alpha :1 - CL
"""

alpha= 1 - 0.95
print(alpha)

"""The standard error:"""

import math
size_y, size =data.shape
sde= sd_y/math.sqrt(size_y)
print(sde)

"""the z value for a confidence level corresponding to 95% is calculated like this:
P(0.95+0.05/2)=0.9750 the z value for probability 0.9750 is 1.96 (using the standard normal table)
"""

z=1.96

"""Now we can calculate the margin of error ME = z * standard error"""

ME= z*sde
print(ME)

"""and finally we can write the expression of the confidence interval :"""

CI=[mean_y-ME, mean_y+ME]
print(CI)

"""**Performing the linear regression :**
the graph of the data  shows that the response follow a sinusoide with respect of time so the true output can be represented like this : y(t)= A + B cos (wt+ phi) = A + B cos (wt) + C sin(wt)

**Estimation of the periode**
"""

from scipy import signal

f, Pxx = signal.periodogram(data.y, fs=250, window='hanning', scaling='spectrum')

plt.figure()
plt.semilogy(f, Pxx, label='PSD')
plt.xlim(0, 5)
plt.grid(color='r', linestyle='--', linewidth=1)
plt.xlabel("Frequency ")
plt.ylabel("Power spectral density ")
plt.legend()
plt.title('Power spectral density as a function of frequency')
plt.show

"""**From the graph we can see that the see that the power spectral density has a pick at f= 0.43 Hz, which correspond to T=2.27**"""

T=2.28

"""**Now we calculate omega**"""

omega=2*np.pi/T
print(omega)

"""**And we perform the regression**"""

size=len(data)
print(size)

A=np.array([np.ones((size)), np.cos(omega*data.x), np.sin(omega*data.x)]).T
print(A)

C=np.array([data.y]).T
print(C)

B=np.dot(np.linalg.pinv(A) , C)
print(B)

"""**And finally we have the estimated response:**"""

y=B[0, 0] + B[1, 0]*np.cos(omega*data.x) + B[2, 0]*np.sin(omega*data.x)

fig = plt.figure() 
plt.plot(data.x, y, label='regression', c='r')
plt.scatter(data.x, data.y,label='original data', c='b')

plt.xlabel("x ")
plt.ylabel("y ")
plt.title("Plot of the linear regression and the original data")
plt.legend()

"""**Task 2 RANSAC**

**First we upload the dataset**
"""

uploaded = files.upload()

import io
data2 = pd.read_csv(io.StringIO(uploaded['data_set_17_.csv'].decode('utf-8')))
print(data2)

data2.columns=['x', 'y', 'z']
print(data2.columns)

"""**We plot the dataset:**"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.view_init(60, 35)

ax.scatter3D(data2.x, data2.y, data2.z,cmap='Greens', label='data');
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.title("Data set")
ax.legend()

"""We can see by rotating the figure that the set of points can be approximated to a plane and since we need 3 points to define a plane so n=3

**Calcualting the number of iterations:**
"""

import math
n=3
w= 0.6
P=0.99
k=math.log(1-P)/math.log(1-math.pow(w, n)) 
print(k)

"""**Now let's estimate the threshold that we will use :**"""

sd_x, sd_y, sd_z=data2.std()
print(sd_x, sd_y, sd_z)

t=3*sd_z
print(t)

"""**Now we create a function to compute the  RANSAC algorithm**

**First let's create a function that return the parameters of a plane from three points **
"""

def plan(p1, p2, p3):
  #A plan has an equation in the form ax+by+cz=d 
  #this function return the parameters a b c d  
  n=np.cross((p3-p1), (p2-p1))
  d=np.dot(n, p1)
  return [n[0],n[1],n[2],d]

p1=np.array([5, 1, 0])
p2=np.array([8, 0, 0])
p3=np.array([5, 1, 1])
print(plan(p1, p2, p3))

def RANSAC(data2, k, threshold, d):  
  iteration=0
  while (iteration<k):
    r=[0, 0, 0]
    m_inlier=1.0*np.array([[0, 0, 0],[0, 0, 0], [0, 0, 0]])
    
    #we randomly select three points in the dataset
    for i in range(3): 
      r[i]= np.random.choice(range(99))
      m_inlier[i]=data2.iloc[r[i]]

    parameters=[0, 0, 0]
    parameters= plan(m_inlier[0, :], m_inlier[1, :], m_inlier[2, :])
    j=0
    i=0    
    n=np.array([parameters[0],parameters[1], parameters[2]])
    inlier=1.0*np.array([0, 0, 0])

    for i in range(99):
      #we verify if the distance between the plane and the point is less the threshold
      if(abs(parameters[0]*data2.x[i]+parameters[1]*data2.y[i]+parameters[2]*data2.z[i])/np.linalg.norm(n)<threshold):
        #np.concatenate((inlier, np.array(data2.iloc[1])), axis=1)    
        j=j+1
    
    if j>d:
      best_parameters=parameters
      print(best_parameters)
      d=j
    iteration=iteration+1
  return best_parameters

#parameters=RANSAC(data2, 50, 8, 10)
parameters=RANSAC(data2, k , t, 10)
print(parameters)

"""**Now lets plot the plane and see if its fits the data:**"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create x,y
x, y = np.meshgrid(np.linspace(-10,10,50), np.linspace(-10,10,50))

# Calculate corresponding z
z = (-parameters[0] * x - parameters[1] * y + parameters[3]) * 1. /parameters[2]

# Plot the surface
ax = plt.figure().gca(projection='3d')
ax.plot_surface(x, y, z, alpha=0.2)

#ax.view_init(60, 35)

#ax = plt.axes(projection='3d')
ax = plt.gca()
ax.hold(True)

#And i would like to plot the dataset : 
ax.scatter3D(data2.x, data2.y, data2.z,cmap='Greens', label='data');
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.title("Dataset and regression plane")
ax.legend()

plt.show()

"""**Let's try with the library skimage.measure**"""

from skimage.measure import LineModelND, ransac

xyz=np.array([np.array(data2.x), np.array(data2.y), np.array(data2.z)])
model_robust, inliers = ransac(xyz, LineModelND, min_samples=2, residual_threshold=10, max_trials=1000)

fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.view_init(60, 35)

ax.scatter3D(data2.x, data2.y, data2.z,cmap='Greens');
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');

ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz[inliers][:, 0], xyz[inliers][:, 1], xyz[inliers][:, 2], c='b',
           marker='o', label='Inlier data')
ax.legend(loc='lower left')
plt.show()

