import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_wine
'''
Built a KNN classifier
'''
data = load_wine()
x = data.data
y = data.target
'''Load Dataset'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = KNN(k=5)
clf.fit(x_train, y_train) #Fit dataset onto model
predictions = clf.predict(x_test) 

accuracy = np.sum(predictions == y_test) / len(y_test)


print("Predictions: " , predictions)
print("Accuracy: " , accuracy)


