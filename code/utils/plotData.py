import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.interpolate import interp1d

from datasets.load_dataset import get_dataset, Datasets
df = pd.read_csv("../datasets/data/ionosphere/ionosphere.data")

X = df.iloc[:, 34].values

y = df.iloc[:,2].values

f = plt.figure()

#For spline data
#xx = np.linspace(X.min(), X.max(), 1000)
#y_smooth = interp1d(X, y)(xx)
#y_smooth = interp1d(x, y, kind="cubic")(xx)

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(xx, y_smooth, "r-")

#For Scatter Plot
plt.scatter(X,y)

plt.show()
#plt.plot(X,y)


#Histogram
#plt.hist(X, alpha= 1, density = 1, facecolor = 'g')


f.savefig("foo.pdf", bbox_inches='tight')


#For counting the occurences for graph.
#(X =='g' ).sum()
#(y=='b').sum()
