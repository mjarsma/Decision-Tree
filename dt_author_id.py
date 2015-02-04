# import sys
from time import time
# sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn import tree;
from sklearn.metrics import accuracy_score;

print len(features_train[0]);

clf40 = tree.DecisionTreeClassifier(min_samples_split=40);

clf40.fit(features_train, labels_train);

pred40 = clf40.predict(features_test);

acc_min_samples_split_40 = accuracy_score(labels_test, pred40);

print acc_min_samples_split_40;