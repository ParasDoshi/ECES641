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
plt.show()
plt.savefig("tsne_plot.png",dpi=1000)


print("Running AgglomerativeClustering algorithm")
knn_graph = kneighbors_graph(X, 30, include_self=False)
for connectivity in (None, knn_graph):
    for n_clusters in (60, 30):
        plt.figure(figsize=(10, 4))
        for index, linkage in enumerate(('average',
                                         'complete',
                                         'ward',
                                         'single')):
            plt.subplot(1, 4, index + 1)
            model = AgglomerativeClustering(linkage=linkage,
                                            connectivity=connectivity,
                                            n_clusters=n_clusters)
            t0 = time.time()
            model.fit(X)
            elapsed_time = time.time() - t0
            plt.scatter(X[:, 0], X[:, 1], c=model.labels_,
                        cmap=plt.cm.nipy_spectral,s=0.4)
            plt.title('linkage=%s\n(time %.2fs)' % (linkage, elapsed_time),
                      fontdict=dict(verticalalignment='top'))
            plt.axis('equal')
            plt.axis('off')

            plt.subplots_adjust(bottom=0, top=.89, wspace=0,
                                left=0, right=1)
            plt.suptitle('n_cluster=%i, connectivity=%r' %
                         (n_clusters, connectivity is not None), size=17)

plt.show()

print("Creating Dendrogram now")
linked = linkage(vectors[:100], 'ward')
labelList=words[:100]
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()
plt.savefig("dendrogram.png",dpi=400)

