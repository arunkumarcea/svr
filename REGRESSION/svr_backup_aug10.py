print(__doc__)
import numpy as np
import scipy as sp
import random
import operator
import csv
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn import datasets,svm, linear_model, cross_validation, grid_search
###############################################################################
# Generate sample data
#with open('b22_500_0.1.txt') as f:
with open('./historyless_and_no_SS/sign_10') as f:
     array=[]
     for line in f:
         array.append([int(x) for x in line.split()])#should be changed to float
X_total=array
X_total=np.asarray(X_total)
#X = np.sort(5 * np.random.rand(40, 1), axis=0)
with open('output_467_avf.txt') as f2:
     array2=[]
     for line in f2:
         array2.append([float(x2) for x2 in line.split()])
y_total=array2
y_total=np.asarray(y_total).ravel()
#y = np.sin(X).ravel()

###############################################################################
# Add noise to targets
#y[::5] += 3 * (0.5 - np.random.rand(8))
###############################################################################
###############################################################################
def MyGridSearch(X,y):
    kfold = cross_validation.KFold(len(X), 5)
    for train, test in kfold:
    	#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5, random_state = 0)
    	#parameters = {'kernel': ('linear', 'rbf'), 'C':[1.5, 10]}
    	#parameters = {'kernel': ['rbf'], 'gamma': [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9 ], 'epsilon' : [0.1],
    	#                 'C': [1, 5, 10, 50,100,500,1000,5000,10000]}
    	#parameters = {'kernel': ['poly'], 'gamma': [1e-2,1e-3,1e-4 ], 'epsilon' : [0.1],'degree':[3],
    	#                 'C': [ 50,100,500,1000]}
    	parameters = {'kernel': ['rbf'], 'gamma': [1e-5], 'epsilon' : [0.2],
                     'C': [100000]}
        #parameters = [{'C': sp.stats.expon(scale=100), 'gamma': sp.stats.expon(scale=.1),
        #             'kernel': ['rbf'], 'class_weight':['auto', None]}]
    	model = svm.SVR()

    	grid = GridSearchCV(model,parameters)
    	#grid = RandomizedSearchCV(model,parameters)
    	grid.fit(X[train], y[train])
	#print grid
    	predictions = grid.predict(X[test])
	print grid.best_score_
	if grid.best_score_ > 0.98:
		return grid
		break
   	#print grid.best_estimator_.coef_
    return grid
    #$%print predictions
    #$%print y[test]
    #$%print len(predictions)
    #$%print len(y[test])
    #$%#grid.best_estimator_.params
    #$%print "Grid best score: ", grid.best_score_
    #$%print "Grid score function: ", grid.score(X[test],y[test])
    #$%print("\tPrecision RBF: %1.3f" % mean_squared_error(y[test], predictions))
    #$%return predictions, y[test]
################################################################################
for x in range(0, 1): #--no point in having this loop. random_state always goes through same values in all iterations
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_total, y_total, test_size=0.3, random_state = random.randrange(0,1000))
    svr_pred=MyGridSearch(X_train,y_train)
    predictions = svr_pred.predict(X_test)
    #checkfit=grid.predict(X_train)
    tau, p_value = sp.stats.kendalltau(predictions,y_test)
    print ("\tTau is: %1.3f " %tau)
    #RRMSE=mean_squared_error(y_test,predictions/y_test)**0.5
    #print ("RRMSE is: %1.3f " %RRMSE)
    if tau > 0.3:
        break
normalised=np.amax(y_test)-np.amin(y_test)
RMSE=mean_squared_error(y_test, predictions)
NRMSE=RMSE*100/normalised
print("\tNRMSE in percent: %1.3f" % NRMSE)
print len(X_train)
div=90/(len(y_test)*1.000)
#print div
array_predict=np.vstack((y_test, predictions))
#print array_predict
arr1inds = y_test.argsort()
sorted_y_test = y_test[arr1inds[::-1]]
sorted_predictions = predictions[arr1inds[::-1]]

with open('predictions.csv', 'w') as f1:
    writer = csv.writer(f1, delimiter='\t')
    writer.writerows(zip(sorted_y_test,sorted_predictions))


#dictA = dict(zip(y_test, predictions))
#sorted_predictions = sorted(dictA.items(), key=operator.itemgetter(0))
#fp = open('predictions.txt','w')
#fp.write('\n'.join('%s %s' % hehe for hehe in sorted_predictions))
#fp.close()
#fp.write('\n'.join('%s %s' % sorted_predictions for sorted_predictions in mylist))
chumma=np.arange(0, 100,div )
chumma=chumma[:len(y_test)]

np.savetxt('predicted_avf.txt', predictions)
np.savetxt('actual_avf.txt', y_test)
#
#f.write(y_test, predictions) # python will convert \n to os.linesep
#f.close()
#fig2 = plt.figure()
#ax1 = fig2.add_subplot(111)
#ax1.plot(chumma, y_test)
#
#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111)
#ax1.plot(chumma, X_test)
# look at the results
#plt.scatter(X, y, c='k', label='data')
plt.hold('on')
plt.plot(chumma, sorted_predictions,'g^', label='RBF model')
#plt.plot(chumma, y_lin, c='r', label='Linear model')
#plt.plot(chumma, y_poly, c='b', label='Polynomial model')
plt.plot(chumma, sorted_y_test, 'rs', label='Original')
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
