import numpy as np
import pandas as pd
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np

read_file=pd.read_csv(r'C:/Users/yaren/OneDrive/Masaüstü/data/iris.data')
read_file.to_csv(r'C:/Users/yaren/OneDrive/Masaüstü/data/datanew.csv', header=["sepal length","sepal width",
                                                                        "petal length","petal width", "class"], index=False)

df=pd.read_csv(r'C:/Users/yaren/OneDrive/Masaüstü/data/datanew.csv', index_col=False)

knn=KNeighborsClassifier()

X=df.loc[:, ["sepal length","sepal width","petal length","petal width"]]
y=df.loc[:, "class"]
param_grid={"n_neighbors": np.arange(1,50)}

knn_cv=GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)
print(knn_cv.best_params_)
print(knn_cv.best_score_)