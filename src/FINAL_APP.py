# import the necessary packages
from skimage.util import img_as_float
from skimage.future import graph
from skimage import io, color
from skimage.metrics import variation_of_information

from sklearn import cluster as cl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize

from statistics import mean, stdev

from utils.node2vec.src import node2vec
from gensim.models import Word2Vec

from os import walk, makedirs 

import networkx as nx
import numpy
import warnings, sys, argparse
warnings.filterwarnings("ignore")

import pickle
import helper
import pandas as pd

if __name__ == "__main__":
    argsy = helper._parse_args()

    methods = { "slic": "SLIC", "msp": "MSP", "mso": "MSO" }
    method = methods[argsy['method']]
    
    # construct the argument parser and parse the arguments

    # meanshift and SLIC arguments
    _spatial_radius=int(argsy['hs']) #hs
    _range_radius=float(argsy['hr']) #hr
    _min_density=int(argsy['mind']) #mind
    _sigma=float(argsy['sigma'])

    # computing best clustering ?
    write = True if argsy['write'] == "True" else False
    best = True if argsy['best'] == "True" else False
    read = True if argsy['read'] == "True" else False
    silh = True if argsy['silhouette'] == "True" else False
    
    path_images = argsy['path']
    
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
    
    # will contain the final PRI and VOI results of every iteration
    GEST_PRI, GEST_VOI = [], []
    dirpath,_,images = list(walk(path_images))[0]

    for i,filename in enumerate(sorted(images)):
        # load the image and convert it to a floating point data type
        image = io.imread(dirpath+filename)
        image = img_as_float(image)
        image_lab = (color.rgb2lab(image) + [0,128,128]) #// [1,1,1]

        # loop over the number of segments
        gt_boundaries, gt_segmentation = helper._get_groundtruth(path_groundtruths+filename[:-4]+".mat")
        if(read):
            Gr = nx.read_gpickle(path_pickles+str(i+1)+"_"+filename[:-4]+".pkl")
            labels = pickle.load(open(path_labels_msp+str(i+1)+"_"+filename[:-4]+".preseg","rb"))
        else:
            labels = helper._meanshift_py(dirpath+filename,_spatial_radius,_range_radius,_min_density)
            Gr = graph.rag_mean_color(image_lab,labels,connectivity=2,mode='similarity',sigma=_sigma)

        number_regions = numpy.amax(labels)
        
        # computing embeddings with Node2vec framework
        Gn2v = node2vec.Graph(Gr, False, 2, .5)
        Gn2v.preprocess_transition_probs()
        walks = Gn2v.simulate_walks(20, 20)
        model=helper.learn_embeddings(walks,dimensions=16)
        
        # getting the embeddings
        representation = model.wv
        gestemb = [representation.get_vector(str(node)).tolist() for node in Gr.nodes()]
        #gestemb = []
        #for l,node in enumerate(Gr.nodes()):
        #    gestemb.append(model.wv.get_vector(str(node)).tolist())
        
        # NOTE: Mean is included in graph somehow?
        feature_vector = normalize(numpy.asarray(helper._color_features(labels,image_lab)))
        for l,v in enumerate(feature_vector):
            gestemb[l].extend(v)
        
        # scaling features
        scaler = StandardScaler()
        datagest = scaler.fit_transform(gestemb)
    
        NUM_CLUSTERS = int(argsy['nclusters']) if not(silh) else min(helper.silhouette(datagest,25),number_regions)
            
        for n_cluster in NUM_CLUSTERS:
            segmentation = numpy.zeros(labels.shape,dtype=int)
    
            # using agglomerative clustering to obtain segmentation 
            clustering = cl.AgglomerativeClustering(n_clusters=n_cluster,affinity='cosine',linkage='average',distance_threshold=None).fit(datagest)
            labels_clustering = clustering.labels_
            # building flat segmentation and then reshaping
            segmentation=numpy.asarray([labels_clustering[value-1]+1 for line in labels for value in line]).reshape(labels.shape)
            
            if(write): 
                pickle.dump(labels,open(path_labels+str(i+1)+"_"+filename[:-4]+".preseg","wb"))
                pickle.dump(segmentation,open(path_labels+str(i+1)+"_"+filename[:-4]+".seg","wb"))
                #pickle.dump(clusterings, open(path_clusterings+str(i+1)+"_"+filename[:-4]+".clt","wb"))
                numpy.save(path_embeddings+filename[:-4]+".emb",gestemb)
                nx.write_gpickle(Gr, path_pickles+str(i+1)+"_"+filename[:-4]+".pkl")
                nx.write_weighted_edgelist(Gr, path_graphs+filename[:-4]+".wgt", delimiter='\t')
                helper._savepreseg(labels, image, path_presegs+filename[:-4]+".png")
                helper._savefig(segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+"_"+str(n_cluster)+".png")
