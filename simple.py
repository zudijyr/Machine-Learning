import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from pandas import read_csv, DataFrame

# import some data to play with
training_data = read_csv('first_10k_training_data.csv', header=0)
#X = training_data.data[:, :2]  # we only take the first two features. We could
#columns = games.columns.tolist()
#columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name"]]
#X = training_data[["feature1", "feature2"]]
X = training_data[training_data.columns[0:21]]  
y = training_data["target"]

h = .01  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
#svc = svm.SVC(kernel='rbf', C=10, gamma=10, probability=True)
svc = svm.SVC(kernel='poly', C=1, gamma=10)
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
#for number in predicted:
#	print(number)
#print(X)
# create a mesh to plot in
#x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#xx, yy, xx2, xx3 = np.meshgrid(np.arange(0, 1, h),
#                     np.arange(0, 1, h),
#                     np.arange(0, 1, h),
#                     np.arange(0, 1, h))
#
## title for the plots
#titles = ['SVC with linear kernel',
#          'LinearSVC (linear kernel)',
#          'SVC with RBF kernel',
#          'SVC with polynomial (degree 3) kernel']
#
#
#for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
#    # Plot the decision boundary. For that, we will assign a color to each
#    # point in the mesh [x_min, x_max]x[y_min, y_max].
#    plt.subplot(2, 2, i + 1)
#    plt.subplots_adjust(wspace=0.4, hspace=0.4)
#
#    Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), xx2.ravel(), xx3.ravel()])
#
#    # Put the result into a color plot
#    Z = Z.reshape(xx.shape)
#    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
#
#    # Plot also the training points
#    #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
#    plt.scatter(X["feature1"], X["feature2"], c=y, cmap=plt.cm.coolwarm)
#    plt.xlabel('Sepal length')
#    plt.ylabel('Sepal width')
#    plt.xlim(xx.min(), xx.max())
#    plt.ylim(yy.min(), yy.max())
#    plt.xticks(())
#    plt.yticks(())
#    plt.title(titles[i])
#
#plt.show()
