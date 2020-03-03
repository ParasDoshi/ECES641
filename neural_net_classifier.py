import pandas as pd 
import numpy as np 
import pickle
import os
import helper_functions as hf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve,recall_score,f1_score,matthews_corrcoef
#import confusion_matrix_pretty_print as cmprint
import csv

def show_metrics(mdl, X, y_true, y_pred):
    probs = mdl.predict_proba(X)




data_dir = '.'
fig_dir = '.'

f = open(os.path.join(data_dir, "X_sample_property.obj"), "rb")
X = pickle.load(f)
f.close()
f = open(os.path.join(data_dir, "y_sample_ibd.obj"), "rb")
y = pickle.load(f)
f.close()
y = list(y[0].values)

# data for classisifaction on otu tables
#f = open(os.path.join(data_dir, "otu_test_.07.obj"), "rb")
#otu_test = pickle.load(f)
#f.close()
#f = open(os.path.join(data_dir, "otu_train_.07.obj"), "rb")
#otu_train = pickle.load(f)
#f.close()
#f = open(os.path.join(data_dir, "map_test_.07.obj"), "rb")
#map_test = pickle.load(f)
#f.close()
#f = open(os.path.join(data_dir, "map_train_.07.obj"), "rb")
#map_train = pickle.load(f)
#f.close()

# creating the train and test dataset
X_train_embedding, X_test_embedding, y_train_embedding, y_test_embedding = train_test_split(X, y, test_size=0.2, random_state=42)

# classifying the data using neural network
# nnet = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
nnet = MLPClassifier()
mdl_embedding = nnet.fit(X_train_embedding, y_train_embedding)

y_predict_embedding = mdl_embedding.predict(X_test_embedding)
acc_score_embedding = accuracy_score(y_test_embedding, y_predict_embedding)
print('score:', acc_score_embedding)

f = plt.figure(figsize=(15,5))
roc_auc, fpr, tpr, average_precision, f1, f2 = hf.computeMLstats(mdl_embedding, X_test_embedding, y_test_embedding, plot=True, plot_pr=True, graph_title = "Neural Net Classifier", flipped = False)
# f.savefig(os.path.join(fig_dir, "neural_net.png"))
print('AUC-ROC', roc_auc)
print('F1', f1)
print('F2', f2)
'''
y_test_csv = [y_test_embedding]
y_predict_csv = [y_predict_embedding]
with open('./y_test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(y_test_csv)
with open('./y_predict.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(y_predict_csv)
'''

'''
# cresting test and train dataset for classification based on otu tables
X_train_otu, X_val, X_test_otu, y_train_otu, y_val, y_test_otu = hf.getMlInput(otu_train, otu_test, map_train, map_test, target = 'IBD', asinNormalized = True)
X_train_otu = pd.concat([X_train_otu, X_val], axis = 0)
y_train_otu = y_train_otu + y_val

nnet = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
mdl_otu = nnet.fit(X_train_otu, y_train_otu)

y_predict_otu = mdl_otu.predict(X_test_otu)
acc_score_otu = accuracy_score(y_test_otu, y_predict_otu)
print('otu score:', acc_score_otu)

f = plt.figure(figsize=(15,5))
roc_auc, fpr, tpr, average_precision, f1, f2 = hf.computeMLstats(mdl_otu, X_test_otu, y_test_otu, plot=True, plot_pr=True, graph_title = "OTU Data Neural Net", flipped = False)
f.savefig(os.path.join(fig_dir, "otu_data_neural_net.png"))

columns = ['IBD', 'Non-IBD']
annot = True;
cmap = 'Oranges';
fmt = '.2f'
lw = 0.5
cbar = False
show_null_values = 2
pred_val_axis = 'y'
#size::
fz = 12;
figsize = [9,9];
if(len(y_test_embedding) > 10):
    fz=9; figsize=[14,14];
conf_mat = cmprint.plot_confusion_matrix_from_data(y_test_embedding, y_predict_embedding, columns,annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)
'''