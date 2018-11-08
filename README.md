# Data Analysis
-----------------------

# importing Numpy, Matplotlib and sklearn libraries
import matplotlib.pyplot as plt
import numpy as np
# importing datasets from scikit-learn
from sklearn import datasets, linear_model
# Load the dataset
house_price = [245, 312, 279, 308, 199, 219, 405, 324, 319, 255]
size = [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]
print(size)
# Reshape the input to your regression
size2 = np.array(size).reshape((-1,1))
print(size2)

# By using fit module in linear regression, user can fit the data frequently and quickly
regr = linear_model.LinearRegression()
regr.fit(size2, house_price)
print("Coefficients: \n" ,regr.coef_)
print("intercept: \n", regr.intercept_)

size_new = 1400
price = (size_new * regr.coef_) + regr.intercept_
print(price)
print(regr.predict([[size_new]]))

# Formula obtained for the trained model
def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x,y)
# Plotting the prediction line
graph('regr.coef_*x+regr.intercept_',range(1000,2700))
plt.scatter(size,house_price, color='black')
plt.ylabel('house price')
plt.xlabel('size of house')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

# Plotting and visualizing our data before feeding it into the Machine Learning Algorithm
x=[1, 5, 1.5, 8, 1, 9] 
y=[2, 8, 1.8, 8, 0.6, 11]
plt.scatter(x,y)
plt.show()

# converting our data to a numpy array
X = np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])
# we initialize K-means algorithm with the required partner and we use .fit() to fit the data
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
# getting the values of centroids and lables based on the fitment
centroids = kmeans.cluster_centers_
labels=kmeans.labels_
print(centroids)
print(labels)

# plotting and visualizing output
colors = ["g.","r.","c.","y."]
for i in range(len(X)):
    print ("coordinate:",X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
plt.scatter(centroids[:,0],centroids[:,1], marker = "x", s=150, linewidths=5, zorder=10)
plt.show()


