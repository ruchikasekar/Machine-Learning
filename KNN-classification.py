
import numpy as np
from collections import Counter
'''Computes Euclidean Distance'''
def euclidian(x1,x2):
  distance = np.sqrt(np.sum((x1-x2)**2))
  return distance

class KNN:
  def __init__(self, k=3):
    self.k = k 
  '''Trains the model with example data'''
  def fit(self,x,y):
    self.x_train = x #x dataset
    self.y_train = y #y dataset
  
  '''Returns the majority label after looping over data points'''
  def predict(self, x):
    predictions = [self.predict_helper(i) for i in x] 
    return predictions
  def predict_helper(self,x):
    #compute distance (euclidian)
    distances = [euclidian(x,x_train) for x_train in self.x_train]

    #get closest K 
    k_indices = np.argsort(distances)[:self.k] #argsort tells us where the indices are after they are sorted and we are getting the indices for the k closest neighbors
    k_closest = [self.y_train[i] for i in k_indices]

    #determine majority vote label
    most_common_label = Counter(k_closest).most_common(1)
    return most_common_label[0][0]
  
    
    
