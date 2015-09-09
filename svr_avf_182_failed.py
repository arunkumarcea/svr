print(__doc__)

import numpy as np
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import datasets,svm, linear_model, cross_validation, grid_search

###############################################################################
# Generate sample data
#with open('b22_500_0.1.txt') as f:
with open('misr_with_no_space_sampling.dat') as f:
     array=[]
     for line in f:
         array.append([int(x) for x in line.split()])
X=array
X=np.asarray(X)
#X = np.sort(5 * np.random.rand(40, 1), axis=0)
with open('output_avf.txt') as f2:
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
###############################################################################
def MyGridSearch(X,y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state = 0)
#    parameters = {'kernel': ('linear', 'rbf'), 'C':[1.5, 10]}
    parameters = {'kernel': ['rbf'], 'gamma': [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9 ], 'epsilon' : [0.1],
                     'C': [1, 5, 10, 50,100,500,1000,5000,10000]}
    #parameters = {'kernel': ['poly'], 'gamma': [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9 ], 'epsilon' : [0.1],'degree':[3],
    #                 'C': [1, 5, 10, 50,100,500,1000,5000,10000]}
    model = svm.SVR()
    grid = GridSearchCV(model,parameters)
    grid.fit(X_train, y_train)
    #print grid.best_estimator_.coef_
    predictions = grid.predict(X_test)
    print predictions
    print y_test
    print len(predictions)
    print len(y_test)
    #grid.best_estimator_.params
    print "Grid best score: ", grid.best_score_
    print "Grid score function: ", grid.score(X_test,y_test)
    print("\tPrecision RBF: %1.3f" % mean_squared_error(y_test, predictions))
    return predictions, y_test
################################################################################
predictions, y_test=MyGridSearch(X,y)


chumma=np.arange(0, 100,0.60 )
chumma=chumma[:len(y_test)]
# look at the results
#plt.scatter(X, y, c='k', label='data')
plt.hold('on')
plt.plot(chumma, predictions,'g', label='RBF model')
#plt.plot(chumma, y_lin, c='r', label='Linear model')
#plt.plot(chumma, y_poly, c='b', label='Polynomial model')
plt.plot(chumma, y_test, 'rs', label='Original')
#plt.plot(chumma, y_rbf, 'r--', chumma, y_lin, 'bs', chumma, y, 'g^', chumma, y_poly, 'y')
#$%plt.plot(chumma, y_rbf, 'r--', label='RBF_model')
#$%plt.plot (chumma, y_lin, 'bs', label='Linear_model')
#$%plt.plot (chumma, y, 'g^', label='actual_value')
#$%plt.plot (chumma, y_poly, 'y', label='Polynomial_model')
plt.xlabel('')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
