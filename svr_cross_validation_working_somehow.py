print(__doc__)

import numpy as np
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import datasets, svm, linear_model, cross_validation, grid_search

###############################################################################
# Generate sample data
with open('b22_500_0.1.txt') as f:
     array=[]
     for line in f:
         array.append([int(x) for x in line.split()])
X=array
X=np.asarray(X)
#X = np.sort(5 * np.random.rand(40, 1), axis=0)
with open('b22_500_CD.txt') as f2:
     array2=[]
     for line in f2:
         array2.append([float(x2) for x2 in line.split()])
y=array2
y=np.asarray(y).ravel()
#y = np.sin(X).ravel()
###############################################################################
# Add noise to targets
#y[::5] += 3 * (0.5 - np.random.rand(8))
###############################################################################
svr=svm.SVR()
##k_fold = cross_validation.KFold(n=100, k=5, indices=True)
##for train_indices, test_indices in k_fold:
kfold = cross_validation.KFold(len(X), 5)
for train, test in kfold:
	svr.fit(X[train], y[train]).score(X[test], y[test])
print (cross_validation.cross_val_score(svr, X, y, cv=kfold, n_jobs=-1))

###############################################################################
# Fit regression model
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#$%svr_lin = SVR(kernel='linear', C=1e3)
#$%svr_poly = SVR(kernel='poly', C=1e3, degree=3)
#$%y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
#$%y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
#$%y_poly = svr_poly.fit(X_train, y_train).predict(X_test)
#$%#$%chumma=np.arange(4000, 4700, 1.35)
#$%#$%chumma=chumma[:500]
#$%print("\tPrecision RBF: %1.3f" % mean_squared_error(y_test, y_rbf))
#$%print("\tPrecision Linear: %1.3f" % mean_squared_error(y_test, y_lin))
#$%print("\tPrecision Polynomial d2: %1.3f" % mean_squared_error(y_test, y_poly))
#$%###############################################################################
#$%# look at the results
#$%#plt.scatter(X, y, c='k', label='data')
#$%#$%plt.hold('on')
#$%#plt.plot(chumma, y_rbf, c='g', label='RBF model')
#$%#plt.plot(chumma, y_lin, c='r', label='Linear model')
#$%#plt.plot(chumma, y_poly, c='b', label='Polynomial model')
#$%#plt.plot(chumma, y, c='y', label='Original')
#$%#plt.plot(chumma, y_rbf, 'r--', chumma, y_lin, 'bs', chumma, y, 'g^', chumma, y_poly, 'y')
#$%#$%plt.plot(chumma, y_rbf, 'r--', label='RBF_model')
#$%#$%plt.plot (chumma, y_lin, 'bs', label='Linear_model')
#$%#$%plt.plot (chumma, y, 'g^', label='actual_value')
#$%#$%plt.plot (chumma, y_poly, 'y', label='Polynomial_model')
#$%#$%plt.xlabel('')
#$%#$%plt.ylabel('target')
#$%#$%plt.title('Support Vector Regression')
#$%#$%plt.legend()
#$%#$%plt.show()
