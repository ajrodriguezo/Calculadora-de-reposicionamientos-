
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import mean_squared_error

Db= pd.read_excel("/content/spss v1.xlsx")
Db.head()

X= [Db.Edad,Db.IMC]
X = np.array(X)
y = Db.I1
y = y.values

X= [Db.Edad,Db.IMC]
y = Db.I1

clf = tree.DecisionTreeRegressor(max_depth=10)
clf = clf.fit(X, y)

y_hat = clf.predict(X)

mean_squared_error(y, y_hat)

y_hat = clf.predict([[55,30]])
y_hat
