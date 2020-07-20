from gensim.models import Word2Vec
 
 
from sklearn import cluster
from sklearn import metrics
 
# training data
 
sentences = [['this', 'is', 'the', 'good', 'machine', 'learning', 'book'],
            ['this', 'is',  'another', 'book'],
            ['one', 'more', 'book'],
            ['this', 'is', 'the', 'new', 'post'],
          ['this', 'is', 'about', 'machine', 'learning', 'post'],  
            ['and', 'this', 'is', 'the', 'last', 'post']]
 
 
# training model
model = Word2Vec(sentences, min_count=1)
 
# get vector data
X = model.wv
print (X.vectors)
 
print (model.similarity('this', 'is'))
 
print (model.similarity('post', 'book'))
 
print (model.most_similar(positive=['machine'], negative=[], topn=2))
 
print (model['the'])
 
NUM_CLUSTERS=3 
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X.vectors)
 
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
 
print ("Cluster id labels for inputted data")
print (labels)
print ("Centroids data")
print (centroids)
 
print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print (kmeans.score(X.vectors))
 
silhouette_score = metrics.silhouette_score(X.vectors, labels, metric='euclidean')
 
print ("Silhouette_score: ")
print (silhouette_score)
