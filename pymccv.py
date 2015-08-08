__author__ = 'hmonfared'
from sklearn.datasets import load_svmlight_file
from sklearn import svm
from sklearn import cross_validation
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils import indexable
import copy
def calc_confusion_table(cfm,label):
    predicted = cfm[label]
    actual    = [cfm[i][label] for i in range(len(cfm))]
    true_pos  = predicted[label]
    false_pos = sum(actual) - true_pos
    false_neg = sum(predicted) - true_pos
    total     = sum([sum(i) for i in cfm])
    true_neg  = total - true_pos - false_pos - false_neg
    return {'TP':true_pos, 'FN':false_neg,'FP':false_pos, 'TN':true_neg}
def get_k_folds(X,Y,k):
    for i in range(k):
        train_indices = [ row for row in range(X.shape[0]) if row % k != i ]
        test_indices = [ row for row in range(X.shape[0]) if row % k == i ]
        yield X[train_indices],Y[train_indices],X[test_indices],Y[test_indices]

def cross_validate(classifier,X,Y,k):        
    avg_measure={'TP':0.0,'FN':0.0,'FP':0.0,'TN':0.0,'PRECISION':0.0,'RECALL':0.0,'ACCURACY':0.0,'FPR':0.0,'TPR':0.0,'F1':0.0}
    for X_train,Y_train,X_test,Y_test in get_k_folds(X,Y,k):
        clsfr = copy.deepcopy(classifier)
        clsfr.fit(X_train,Y_train)
        pred = clsfr.predict(X_test)
        print 'accuracy :' , np.mean( Y_test == pred )
        conf_mat = confusion_matrix(Y_test, pred)
        avg_true_pos, avg_false_neg,avg_false_pos, avg_true_neg = 0.0,0.0,0.0,0.0
        class_len = len(conf_mat[0])
        #print conf_mat
        for class_idx in range(class_len):
            ct = calc_confusion_table(conf_mat,class_idx)
            avg_true_pos += ct['TP']/float(class_len)
            avg_false_neg += ct['FN']/float(class_len)
            avg_false_pos += ct['FP']/float(class_len)
            avg_true_neg += ct['TN']/float(class_len)
        print avg_true_pos,avg_false_neg,avg_false_pos,avg_true_neg
        avg_measure['TP'] += avg_true_pos/k
        avg_measure['FN'] += avg_false_neg/k
        avg_measure['FP'] += avg_false_pos/k
        avg_measure['TN'] += avg_true_neg/k
        prec = avg_true_pos/(avg_true_pos+avg_false_pos)
        avg_measure['PRECISION'] += prec/k
        rec = avg_true_pos/(avg_true_pos+avg_false_neg) 
        avg_measure['RECALL'] += rec/k
        accu = (avg_true_pos+avg_true_neg)/(avg_true_pos+avg_true_neg+avg_false_pos+avg_false_neg)
        print 'accccuu :', accu,'prec :',prec,'recall:',rec
        avg_measure['ACCURACY'] += accu/k
        avg_measure['FPR'] += avg_false_pos/(avg_true_pos+avg_true_neg+avg_false_pos+avg_false_neg)/k
        avg_measure['TPR'] += avg_true_pos/(avg_true_pos+avg_true_neg+avg_false_pos+avg_false_neg)/k
        avg_measure['F1'] += (2*prec*rec/(prec+rec) if prec>0 and rec >0 else 0)/k
    return avg_measure
        
        
        
