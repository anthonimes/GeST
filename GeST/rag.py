# import the necessary packages
from skimage.util import img_as_float
from skimage.future import graph
from skimage import io, color
from skimage.metrics import variation_of_information

from sklearn import cluster as cl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import normalize

from statistics import mean, stdev
from math import log

from utils.node2vec.src import node2vec
from gensim.models import Word2Vec

from os import walk, makedirs 
from matplotlib import pyplot as plt

import networkx as nx
import numpy, time
import warnings, sys, argparse
warnings.filterwarnings("ignore")

import pickle
import helper

def silhouette(points,kmax):
    def SSE():
        sse=[]
        for k in range(2, kmax):
            km = cl.AgglomerativeClustering(n_clusters=k,affinity='cosine',linkage='average',distance_threshold=None).fit(points)
            labels_clustering = km.labels_
            silhouette_avg=silhouette_score(points, labels_clustering, metric = 'cosine')
            sse.append(silhouette_avg)
        return sse
        
    scores = SSE()
    best = scores.index(max(scores))+2
    return best

# FIXME: NEW ATTEMPT --- select best number of clusters according to 3 scores, and ouput mean
def allthree(points,kmax):
    def scores():
        sse,ch,db=[],[],[]
        for k in range(2, kmax):
            km = cl.AgglomerativeClustering(n_clusters=k,affinity='cosine',linkage='average',distance_threshold=None).fit(points)
            labels_clustering = km.labels_  
            tmp=silhouette_score(points, labels_clustering, metric = 'cosine')
            sse.append(tmp)
            tmp=calinski_harabasz_score(points,labels_clustering)
            ch.append(tmp)
            tmp=davies_bouldin_score(points, labels_clustering)
            db.append(tmp)
        return [sse,ch,db]
    scores = scores()
    best = int(mean([scores[0].index(max(scores[0]))+2,scores[2].index(min(scores[2]))+2]))
    #best = int(mean([scores[0].index(max(scores[0]))+2,scores[1].index(max(scores[1]))+2,scores[2].index(min(scores[2]))+2]))
    return best

def learn_embeddings(walks,dimensions=32,window_size=5,min_count=0,workers=4,iter=1):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=min_count, sg=1, workers=workers, iter=iter)
    return model

if __name__ == "__main__":
    argsy = helper._parse_args()

    methods = { "slic": "SLIC", "msp": "MSP", "mso": "MSO" }
    which_folder = {"val": "val/", "train": "train/", "test": "test/", "bug": "bug/", "hard_msp": "hard_msp/", "from_observation": "from_observation/"}
    folder = which_folder[argsy['dataset']]
    method = methods[argsy['method']]
    
    # construct the argument parser and parse the arguments

    # meanshift and SLIC arguments
    _spatial_radius=int(argsy['hs']) #hs
    _range_radius=float(argsy['hr']) #hr
    _min_density=int(argsy['mind']) #mind
    _sigma=float(argsy['sigma'])

    # computing best clustering ?
    write = True if argsy['write'] == "True" else False
    read = True if argsy['read'] == "True" else False
    silh = True if argsy['silhouette'] == "True" else False
    
    if(not(silh)):
        n_cluster = int(argsy['nclusters'])
    
    path_images = argsy['path']+"/images/"+folder
    path_groundtruths = argsy['path']+"/groundTruth/"+folder
    
    if method == "SLIC":
        common=method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"+folder
    else:
        common=method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder
      
    path_graphs = "results/graphs/"+common
    path_pickles = "results/pickles/"+common
    path_labels_msp = "results/labels/"+common
    path_labels = "results/labels/"+common
    path_scores = "results/scores/"+common
    path_figs = "results/figs/"+common
    path_presegs = "results/presegs/"+common
    path_embeddings = "results/embeddings/"+common
    path_clusterings = "results/clusterings/"+common
    path_matlab = "results/matlab/"+common

    makedirs(path_graphs,exist_ok=True)
    makedirs(path_pickles,exist_ok=True)
    makedirs(path_labels,exist_ok=True)
    makedirs(path_scores,exist_ok=True)
    makedirs(path_figs,exist_ok=True)
    makedirs(path_presegs,exist_ok=True)
    makedirs(path_embeddings,exist_ok=True)
    makedirs(path_clusterings,exist_ok=True)
    makedirs(path_matlab,exist_ok=True)
    
    path_merge = argsy['path']+"/images/from_observation/"
    _, _, hardimages = list(walk(path_merge))[0]

    # will contain the final PRI and VOI results of every iteration
    GEST_PRI, GEST_VOI = [], []
    GEST_PRI_AT, GEST_VOI_AT = [], []
    dirpath,_,images = list(walk(path_images))[0]

    for _thr in range(10):
        PRI, VOI = [], []
        PRIAT, VOIAT = [], []
        for i,filename in enumerate(sorted(images)):
            # load the image and convert it to a floating point data type
            debut=time.time()
            image = io.imread(dirpath+filename)
            image = img_as_float(image)
            image_lab = (color.rgb2lab(image) + [0,128,128]) #// [1,1,1]
            end=time.time()
            #print("image loaded in {} seconds".format(end-debut))

            # loop over the number of segments
            gt_boundaries, gt_segmentation = helper._get_groundtruth(path_groundtruths+filename[:-4]+".mat")
            if(read):
                Gr = nx.read_gpickle(path_pickles+str(i+1)+"_"+filename[:-4]+".pkl")
                labels = pickle.load(open(path_labels_msp+str(i+1)+"_"+filename[:-4]+".preseg","rb"))
            else:
                labels = helper._meanshift_py(dirpath+filename,_spatial_radius,_range_radius,_min_density)
                fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))
                Gr = graph.rag_mean_color(image_lab,labels,connectivity=2,mode='similarity',sigma=_sigma)
                lc = graph.show_rag(labels,Gr,image)
                cbar = plt.colorbar(lc)
                fig.colorbar(lc, fraction=0.03, ax=ax[0])
                plt.show()
