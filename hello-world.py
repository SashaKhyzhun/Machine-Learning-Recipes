# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import tree

# 0 is bumpy, 1 is smooth
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

# 0 for apple, 1 for oragne
labels = [0, 0, 1, 1]


# decision tree (classifier)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

prediction = clf.predict([[140, 0]])
print(prediction)