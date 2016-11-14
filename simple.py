import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from pandas import read_csv, DataFrame

# import some data to play with
training_data = read_csv('first_10k_training_data.csv', header=0)
X = training_data[training_data.columns[0:21]]  
y = training_data["target"]

h = .01  # step size in the mesh

#svc = svm.SVC(kernel='poly', C=10, gamma=10, probability=True)
svc = svm.SVC(kernel='rbf', C=10, gamma=100, probability=True, cache_size=1000)
svc.fit(X, y)
score = svc.score(X,y)

print(score)
tournament_data = read_csv('first_1000_training_data.csv', header=0)
#tournament_data = read_csv('small_tournament_data.csv', header=0)
x_test = tournament_data[tournament_data.columns[0:21]]  
#x_test = tournament_data[tournament_data.columns[1:22]]
#predicted = svc.predict_proba(x_test)
predicted = svc.predict(x_test)
#predicted.insert(0, tournament_data[0])
#df = DataFrame(predicted,index=tournament_data[0])
df = DataFrame(predicted)
df.to_csv('out.csv')
