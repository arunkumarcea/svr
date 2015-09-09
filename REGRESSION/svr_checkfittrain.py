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
from math import sqrt
###############################################################################
#signature_file='./historyless_and_no_SS/sign_10'
#signature_file='./ff_icache/leon3mpfficache.dat'
#signature_file='../b22_500_0.1.txt'
signature_file='sppo.txt'
#signature_file='./2200_workloads/signature_2200_100.dat'
avf_file='output_467_avf.txt'
#avf_file='../b22_500_CD.txt'
#avf_file='./2200_workloads/output_avf_2200_sha.txt'
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
##################### deleting outliers #######################################
#for idx,val in np.ndenumerate(y_total):
#  if y_total[idx] > 0.8:
#    y_total_new=np.delete(y_total, idx)
#    X_total_new=np.delete(X_total, idx)
#X_total=X_total_new
#y_total=y_total_new
#print len(X_total)
#print len(y_total)
###############################################################################
###############################################################################
###############################################################################
def MyGridSearch(X,y):
    #parameters = {'kernel': ('linear', 'rbf'), 'C':[1.5, 10]}
    #parameters = {'kernel': ['rbf'], 'gamma': [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9 ], 'epsilon' : [0.1],
    #                 'C': [1, 5, 10, 50,100,500,1000,5000,10000]}
    #parameters = {'kernel': ['poly'], 'gamma': [1e-2,1e-3,1e-4 ], 'epsilon' : [0.1],'degree':[3],
    #                 'C': [ 50,100,500,1000]}
    #C_range = np.array([1000,2000])
    #gamma_range = np.array([1e-2,1e-1,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9])
    #gamma_range = np.array([1e-4,1e-5,1e-6])
    C_range = np.linspace(1000,100000,20 )
    gamma_range = np.linspace(1e-2, 1e-4, 3)
    epsilon_range = np.array([0.1])

    #epsilon_range = np.linspace(0.1,0.001,10)
    parameters = {'kernel': ['rbf'], 'gamma': gamma_range, 'epsilon' : epsilon_range,
                 'C': C_range}
    #parameters = [{'C': sp.stats.expon(scale=100), 'gamma': sp.stats.expon(scale=.1),
    #             'kernel': ['rbf'], 'class_weight':['auto', None]}]
    model = svm.SVR(verbose=1)

    grid = GridSearchCV(model,parameters,cv=10,verbose=10,n_jobs=4)

    grid.fit(X, y)

    print grid.best_params_
    predictions = grid.predict(X)
    print grid.best_score_
    return grid
################################################################################
for loop_iter in range(0, 1): #
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_total, y_total, test_size=0.3, random_state = random.randrange(0,1000))
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_total, y_total, test_size=0.3, random_state = 42)
    svr_pred=MyGridSearch(X_train,y_train)
    predictions = svr_pred.predict(X_test)
    checkfit=svr_pred.predict(X_train)
    tau, p_value = sp.stats.kendalltau(predictions,y_test)
    print ("\tTau is: %1.3f " %tau)
    if tau > 0.3:
        break
normalised=np.amax(y_test)-np.amin(y_test)
RMSE=sqrt(mean_squared_error(y_test, predictions))
NRMSE=RMSE*100/normalised
print("\tNRMSE in percent: %1.3f" % NRMSE)
print len(X_train)
div=90/(len(y_train)*1.000)

array_predict=np.vstack((y_train, checkfit))

arr1inds = y_train.argsort()
sorted_y_train = y_train[arr1inds[::-1]]
sorted_checkfit = checkfit[arr1inds[::-1]]

with open('checkfit.csv', 'w') as f1:
    writer = csv.writer(f1, delimiter='\t')
    writer.writerows(zip(sorted_y_train,sorted_checkfit))


chumma=np.arange(0, 100,div )
chumma=chumma[:len(y_train)]

np.savetxt('predicted_avf.txt', checkfit)
np.savetxt('actual_avf.txt', y_test)

plt.hold('on')
plt.plot(chumma, sorted_checkfit,'g^', label='RBF model')

plt.plot(chumma, sorted_y_train, 'rs', label='Original')

plt.xlabel('')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
