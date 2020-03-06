import os.path
import pickle
import time
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.cluster.hierarchy import dendrogram, linkage



if os.path.isfile('tsne_output.pkl'):
    print("tsne_output.pkl exists so loading file")
    with open('tsne_output.pkl','rb') as f:  # Python 3: open(..., 'rb')
        words,vectors,X = pickle.load(f)
else:
    embeddings_dict = {}
    with open("vectors.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    print("tsne_output.pkl does not exist making file this should take around 5 mins")
    tsne = TSNE(n_components=2, random_state=0)
    words =  list(embeddings_dict.keys())
    vectors = [embeddings_dict[word] for word in words]
    X = tsne.fit_transform(vectors)
    #save file 
    with open('tsne_output.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([words,vectors,X], f)

plt.scatter(X[:, 0], X[:, 1], s=0.4)
plt.title("Scatter Plot of Processed GloVe Vectors t-SNE")
plt.savefig("tsne_plot.png",dpi=1000)
plt.show()


print("Creating Dendrogram now")
linked = linkage(vectors[:1000], 'ward')
labelList=words[:1000]
fig = plt.figure(figsize=(50,50))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.savefig("dendrogram.png",dpi=500)
plt.show()
