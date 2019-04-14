#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:47:44 2019

@author: SashaKhyzhun
"""

import numpy as np
import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO

# load the dataset
iris = load_iris()

# idk
test_idx = [0, 50, 100]

#  training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data 

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# create decision tree classifier and train in it on the testing data
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# Predict label for new flower
print(test_target)
print(clf.predict(test_data))

# Visualize the tree


# import graphviz

dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True,
                     rounded=True,
                     impurity=False
                     )
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

# graph = graphviz.Source(dot_data.getvalue())
# graph.render("iris.pdf", view=True)
