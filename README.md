# PyMCCV
Python multi class cross validator
This is a simple Python utility for evaluating your classifier using K-Fold cross validation.
When your dataset has more than two classes, measuring your classifier performance based on some scoring methods will be difficult.
I utilized confusion matrix and confusion table to solve this issue.
This library calculates Accuracy, Precision, Recall, TPR, FPR and F1-Score of your classifier with K-Fold Cross Validation.

X,Y=load_svmlight_file('output/vsm_idf.libsvm')
clf = svm.SVC(kernel='rbf', C=2)
cross_validate(clf,X,Y,10)
