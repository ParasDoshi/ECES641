import pandas as pd 
import numpy as np 
import pickle
import os
import helper_functions as hf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
import csv
import scikitplot as skplt

data_dir = '.'
fig_dir = '.'

# data for classisifaction on embeddings based on sample
f = open("X_sample_property.obj", "rb")
X = pickle.load(f)
print(X)
f.close()
f = open("y_sample_ibd.obj", "rb")
y = pickle.load(f)
print(y)
f.close()
y = list(y[0].values)

# data for classisifaction on otu tables
# f = open(os.path.join(data_dir, "otu_test_.07.obj"), "rb")
# otu_test = pickle.load(f)
# f.close()
# f = open(os.path.join(data_dir, "otu_train_.07.obj"), "rb")
# otu_train = pickle.load(f)
# f.close()
# f = open(os.path.join(data_dir, "map_test_.07.obj"), "rb")
# map_test = pickle.load(f)
# f.close()
# f = open(os.path.join(data_dir, "map_train_.07.obj"), "rb")
# map_train = pickle.load(f)
# f.close()

# creating the train and test dataset
X_train_embedding, X_test_embedding, y_train_embedding, y_test_embedding = train_test_split(X, y, test_size=0.2, random_state=42)

# classifying the data using KNN
neigh =  KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski', metric_params=None, n_jobs=-1, n_neighbors=1, p=2, weights='uniform')
mdl_embedding = neigh.fit(X_train_embedding, y_train_embedding)
# mdl_otu = neigh.fit(X_train_otu, y_train_otu)

from sklearn.metrics import accuracy_score
y_predict_embedding = mdl_embedding.predict(X_test_embedding)
acc_score_embedding = accuracy_score(y_test_embedding, y_predict_embedding)
print('embedding score:', acc_score_embedding)

y_test_csv = [y_test_embedding]
y_predict_csv = [y_predict_embedding]
#with open('C:/Users/mk344/OneDrive - Drexel University/Drexel courses/Fall Quarter 2019-2020/ECEC 487/microbiome_glove_embedding/y_test_knn_f2.csv', 'w') as f:
#    writer = csv.writer(f)
#    writer.writerows(y_test_csv)
#with open('C:/Users/mk344/OneDrive - Drexel University/Drexel courses/Fall Quarter 2019-2020/ECEC 487/microbiome_glove_embedding/y_predict_knn_f2.csv', 'w') as f:
#    writer = csv.writer(f)
#    writer.writerows(y_predict_csv)

f = plt.figure(figsize=(15,5))
roc_auc, fpr, tpr, average_precision, f1, f2 = hf.computeMLstats(mdl_embedding, X_test_embedding, y_test_embedding, plot=True, plot_pr=True, flipped = False)
print('F1', f1, 'F2', f2)
f.savefig(os.path.join(fig_dir, "embed_data_knn_classifier_k5_f2.png"))
