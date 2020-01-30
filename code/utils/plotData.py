import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.interpolate import interp1d

from datasets.load_dataset import get_dataset, Datasets

df = pd.read_csv("../datasets/data/breast-cancer-wisconsin/breast-cancer-wisconsin.data")

X = df.iloc[:, 1].values

y = df.iloc[:,2].values

#For spline data
xx = np.linspace(X.min(), X.max(), 1000)
y_smooth = interp1d(X, y)(xx)
#y_smooth = interp1d(x, y, kind="cubic")(xx)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xx, y_smooth, "r-")


plt.scatter(X,y)

plt.show()
#plt.plot(X,y)


#Histogram
#plt.hist(X, alpha= 1, density = 1, facecolor = 'g')

plt.scatter(X,y)
#Scatter plot data

