########################## DECISION TREE #################################
### Two decision tree classifiers with min_sample_split of 2 and 50:

from sklearn import tree;
from sklearn.metrics import accuracy_score;

clf2 = tree.DecisionTreeClassifier(min_samples_split=2);
clf50 = tree.DecisionTreeClassifier(min_samples_split=50);

clf2.fit(features_train, labels_train);
clf50.fit(features_train, labels_train);

pred2 = clf2.predict(features_test);
pred50 = clf50.predict(features_test);

acc_min_samples_split_2 = accuracy_score(labels_test, pred2);
acc_min_samples_split_50 = accuracy_score(labels_test, pred50);

print acc_min_samples_split_2;
print acc_min_samples_split_50;