import pandas as pd
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

read_file=pd.read_csv(r'C:/Users/yaren/OneDrive/Masaüstü/data/iris.data')
read_file.to_csv(r'C:/Users/yaren/OneDrive/Masaüstü/data/datanew.csv', header=["sepal length","sepal width",
                                                                        "petal length","petal width", "class"], index=False)

df=pd.read_csv(r'C:/Users/yaren/OneDrive/Masaüstü/data/datanew.csv', index_col=False)

X=df.loc[:, ["sepal length","sepal width","petal length","petal width"]]
y=df.loc[:, "class"]

knn=KNeighborsClassifier(n_neighbors=6)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=21)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
print("test set predictions: {}".format(y_pred))
print(knn.score(X_test, y_test))

