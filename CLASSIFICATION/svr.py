#!/usr/bin/env python
print(__doc__)
import numpy as np
import scipy as sp
import random
import operator
import csv
import sys
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn import datasets,svm, linear_model, cross_validation, grid_search
from math import sqrt
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
###############################################################################
#signature_file='../results/historyless_and_no_SS/sign_10'
#signature_file='../results/ff_icache/leon3mpfficache.dat'
#signature_file='../b22_500_0.1.txt'
signature_file='../sppo_10.txt'
#signature_file='../results/2200_workloads/signature_2200_100.dat'
#signature_file='../misr_with_no_space_sampling.dat'
#avf_file='../results/output_467_avf.txt'
#avf_file='../output_avf.txt'
avf_file='output_467_avf.txt'
#avf_file='../b22_500_CD.txt'
#avf_file='../results/2200_workloads/output_avf_2200_sha.txt'
with open(signature_file) as f:
     array=[]
     for line in f:
         array.append([float(x) for x in line.split()])#should be changed to float
X_total=array
X_total=np.asarray(X_total)

with open(avf_file) as f2:
     array2=[]
     for line in f2:
         array2.append([float(x2) for x2 in line.split()])
y_total=array2
y_total=np.asarray(y_total).ravel()
print "read_files"
print y_total
for iter_array in range(len(y_total)):
    if 0<=y_total[iter_array]<=0.2:
        y_total[iter_array]=0
    elif 0.2<y_total[iter_array]<=0.4:
        y_total[iter_array]=1
    elif 0.4<y_total[iter_array]<=0.6:
        y_total[iter_array]=2
    elif 0.6<y_total[iter_array]<=0.8:
        y_total[iter_array]=3
    elif 0.8<y_total[iter_array]<=1:
        y_total[iter_array]=4
    else:
	print y_total[iter_array]
print y_total
X_total = StandardScaler().fit_transform(X_total)
##X_total_scaled=preprocessing.scale(X_total)
##X_total=X_total_scaled
##scaler = preprocessing.StandardScaler().fit(X_total)
#svc = SVC(kernel="linear")
## The "accuracy" scoring is proportional to the number of correct
## classifications
#rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y_total, 2),
#              scoring='accuracy')
#rfecv.fit(X_total, y_total)
#
#print("Optimal number of features : %d" % rfecv.n_features_)
#
## Plot number of features VS. cross-validation scores
#plt.figure()
#plt.xlabel("Number of features selected")
#plt.ylabel("Cross validation score (nb of correct classifications)")
#plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#plt.show()
#sys.exit() ###REMOVEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
###############################################################################
###############################################################################
###############################################################################
def MyGridSearch(X,y):
    #parameters = {'kernel': ('linear', 'rbf'), 'C':[1.5, 10]}
    #parameters = {'kernel': ['rbf'], 'gamma': [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9 ], 'epsilon' : [0.1],
    #                 'C': [1, 5, 10, 50,100,500,1000,5000,10000]}
    #parameters = {'kernel': ['poly'], 'gamma': [1e-2,1e-3,1e-4 ], 'epsilon' : [0.1],'degree':[3],
    #                 'C': [ 50,100,500,1000]}
    C_range = np.array([100,1000,10000])
    #gamma_range = np.array([1e-2,1e-1,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9])
    gamma_range = np.array([1e-3,1e-4,1e-5,1e-6])
    #C_range = np.linspace(1e2,1e4,5 )
    #gamma_range = np.linspace(1e-3, 1e-5, 5)
    epsilon_range = np.array([0.1])
    parameters = {'kernel': ['rbf'], 'gamma': gamma_range,
                 'C': C_range}
    #parameters = [{'C': sp.stats.expon(scale=100), 'gamma': sp.stats.expon(scale=.1),
    #             'kernel': ['rbf'], 'class_weight':['auto', None]}]
    model = svm.SVC(verbose=10)

    grid = GridSearchCV(model,parameters,cv=15,verbose=1000,n_jobs=4)

    grid.fit(X, y)

    print grid.best_params_
    predictions = grid.predict(X)
    print grid.best_score_
    return grid
################################################################################
for loop_iter in range(0, 1): #
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_total, y_total, test_size=0.3, random_state = random.randrange(0,1000))
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_total, y_total, test_size=0.3, random_state = 42)
    svr_pred=MyGridSearch(X_train,y_train)
    predictions = svr_pred.predict(X_test)
    print "accuracy"
    print accuracy_score(y_test, predictions)
np.savetxt('predicted_avf.txt', predictions)
np.savetxt('actual_avf.txt', y_test)
    #checkfit=svr_pred.predict(X_train)
#    tau, p_value = sp.stats.kendalltau(predictions,y_test)
#    print ("\tTau is: %1.3f " %tau)
#    if tau > 0.3:
#        break
#normalised=np.amax(y_test)-np.amin(y_test)
#RMSE=sqrt(mean_squared_error(y_test, predictions))
#NRMSE=RMSE*100/normalised
#print("\tNRMSE in percent: %1.3f" % NRMSE)
print len(X_train)
div=90/(len(y_test)*1.000)

array_predict=np.vstack((y_test, predictions))

arr1inds = y_test.argsort()
sorted_y_test = y_test[arr1inds[::-1]]
sorted_predictions = predictions[arr1inds[::-1]]

with open('predictions.csv', 'w') as f1:
    writer = csv.writer(f1, delimiter='\t')
    writer.writerows(zip(sorted_y_test,sorted_predictions))


chumma=np.arange(0, 100,div )
chumma=chumma[:len(y_test)]

np.savetxt('predicted_avf.txt', predictions)
np.savetxt('actual_avf.txt', y_test)

plt.hold('on')
plt.plot(chumma, sorted_predictions,'g^', label='RBF model')

plt.plot(chumma, sorted_y_test, 'rs', label='Original')

plt.xlabel('')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
