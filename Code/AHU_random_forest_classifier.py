# to-do --> document results of this test run with all relevant parameters

import os
import pandas as pd
import numpy as np
import pydot
import itertools
from sklearn.model_selection import train_test_split
from pyts.classification import TimeSeriesForest
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz

data = pd.read_csv(r'/home/vaibhavs/Master_Thesis/ma-vaibhav/Data/data.csv')

# convert dataframe format into (n-samples,n-timestamps)
all_points = []
all_labels = []
datapoint_names = data.columns.tolist()
datapoint_names.remove('time')

for datapoint in datapoint_names:
    data_list = []
    label_list = []
    dp_timeseries = data[['time',datapoint]]
    rows = dp_timeseries.shape[0]
    for i in range(0,rows,193):
        if i <= rows-192:
            sample = dp_timeseries.iloc[i:i+192, [1]].transpose()
            sample_list = sample.values.tolist()
            data_list.append(sample_list[0])
    
    label_list = [datapoint] * len(data_list)
    all_labels.append(label_list) 
    all_points.append(data_list)

all_points = list(itertools.chain.from_iterable(all_points))
all_labels = list(itertools.chain.from_iterable(all_labels))

# check if number of labels (y) and number of time series (x) is equal
assert len(all_points) == len(all_labels), f"length of timeseries values list ({len(all_points)}) is not equal to length of datapoint labels list ({len(all_labels)}). Please make sure timeseries and their datapoint labels are of equal length"

numpy_data = np.asarray(all_points)
numpy_labels = np.asarray(all_labels)

# split data aand labels into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(numpy_data, numpy_labels, test_size=0.20, random_state=42, shuffle=True)
print(X_train.shape)

# fit a Random Forest classifier to the train dataset
clf = TimeSeriesForest(n_estimators=1,
                       n_windows=1,
                       random_state=43, 
                        class_weight="balanced",
                        max_leaf_nodes=15,
                        max_depth=6)
clf.fit(X_train, y_train)
print("Accuracy on test set: ",clf.score(X_test, y_test))
print(clf.feature_importances_)
print(clf.n_features_in_)

feature_names = ['mean','std','slope']
# fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
# tree.plot_tree(clf.estimators_[0],
#                feature_names = feature_names, 
#                class_names=datapoint_names,
#                filled = True);
# fig.savefig('rf_individualtree.png')

# visualize a decision tree from the random forest
export_graphviz(clf.estimators_[0],out_file='tree.dot',
    filled=True,
    rounded=True,
    feature_names=feature_names)
(graph,) = pydot.graph_from_dot_file('tree.dot')
name = 'tree'
graph.write_png(name+  '.png')
os.system('dot -Tpng tree.dot -o tree.png')

# predict class for test samples
test_sample = X_test[:5]
test_truth = y_test[:5]
classes_pred = clf.predict(test_sample)
print("Predicted: ", classes_pred)
print("True :", test_truth)