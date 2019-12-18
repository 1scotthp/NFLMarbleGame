# Python version

import sys

import clf as clf

print('Python: {}'.format(sys.version))
# scipy
import scipy

print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy

print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib

print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas

print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn

print('sklearn: {}'.format(sklearn.__version__))

# Load libraries
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn import linear_model
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

import pandas as pd

print("Hello World")
names = ['homePercent', 'roadPercent', 'GOOD_LOSS_REWARD', 'MAX_WIN_MARGIN', 'MARGIN_WEIGHT', 'correctPercent']
dataset = pd.read_csv("DATA.csv", names=names)

# dataset.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
# print(dataset.groupby('correctPercent').size())

array = dataset.values
X = array[:, 0:5]
Y = array[:, 5]
Z = array[:,0:1]
A = array[:,1:2]
B = array[:,3:4]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20)

# Spot Check Algorithms
models = []
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', svm.SVR()))
models.append(('BAYE', linear_model.BayesianRidge()))
models.append(('Lasso', linear_model.LassoLars()))
models.append(('PASIV', linear_model.PassiveAggressiveRegressor())),
models.append(('RFR', RandomForestRegressor()))
# evaluate each model in turn
results = []
names = []

colors = (1,0,0)
area = 4
plt.scatter(Z, Y, s=area, c=colors, alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(A, B, Y, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('correct')

plt.show()

for name, model in models:
 	kfold = StratifiedKFold(n_splits=10)
 	#clf = svm.SVC(kernel='linear').fit(X_train, Y_train)
	clf = model;
	clf.fit(X_train, Y_train)
	#new_scorer = make_scorer(clf.score(X_validation, Y_validation))

 	cv_results = cross_val_score(model, X_train, Y_train, cv=10)
 	#cv_results = clf.score(X_validation, Y_validation)
	results.append(cv_results)
 	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
	if(name == 'RFR'):
		feature_importances = clf.feature_importances_
		print (feature_importances)


	pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()
