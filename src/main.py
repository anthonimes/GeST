# import the necessary packages
from skimage.util import img_as_float
from skimage.future import graph
from skimage import io, color

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from utils.node2vec.src import node2vec

from os import walk, makedirs 

import networkx as nx
import numpy
import warnings
warnings.filterwarnings("ignore")

import pickle
import helper
#import pandas as pd

if __name__ == "__main__":
    
    # construct the argument parser and parse the arguments
    argsy = helper._parse_args()

    methods = { "slic": "SLIC", "msp": "MSP", "mso": "MSO" }
    method = methods[argsy['method']]

    # common arguments
    _sigma=float(argsy['sigma'])

    # computing best clustering ?
    write = True if argsy['write'] == "True" else False
    silh = True if argsy['silhouette'] == "True" else False
    merge = True if argsy['merge'] == "True" else False
   
    # TODO: allow for a single image or for path
    path_images = argsy['path']
    
    # meanshift and SLIC arguments
    if method == "SLIC":
        _num_segments = float(argsy['segments'])
        _compactness = float(argsy['compactness'])
        common=method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"
    else:
        _spatial_radius=int(argsy['hs']) #hs
        _range_radius=float(argsy['hr']) #hr
        _min_density=int(argsy['mind']) #mind
        common=method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"
     
    path_segmentation = "results/segmentation/"+common
    makedirs(path_segmentation,exist_ok=True)

    # will contain the final PRI and VOI results of every iteration
    GEST_PRI, GEST_VOI = [], []
    dirpath,_,images = list(walk(path_images))[0]

    for i,filename in enumerate(sorted(images)):
        # load the image and convert it to a floating point data type
        image = io.imread(dirpath+filename)
        image = img_as_float(image)
        image_lab = (color.rgb2lab(image) + [0,128,128]) #// [1,1,1]

        # loop over the number of segments
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
        embeddings = [representation.get_vector(str(node)).tolist() for node in Gr.nodes()]
        # NOTE: Mean is included in graph somehow?
        feature_vector = normalize(numpy.asarray(helper._color_features(labels,image_lab)))
        for l,v in enumerate(feature_vector):
            embeddings[l].extend(v)
        
        # scaling features
        scaler = StandardScaler()
        datagest = scaler.fit_transform(embeddings)
        n_cluster = int(argsy['nclusters']) if not(silh) else min(helper.silhouette(datagest,25),number_regions)
        
        clustering, segmentation = helper.GeST(embeddings, labels, n_cluster)
        if(merge):
            segmentation,has_merged=helper._merge(segmentation,image_lab,thr_pixels=250,thr=0.998,sigma=_sigma)
        helper._savefig(segmentation, image, path_segmentation+str(i+1)+"_"+filename[:-4]+"_"+str(n_cluster)+".png")

        if(write): 
            # TODO, define this in helper
            path_graphs = "results/graphs/"+common
            path_pickles = "results/pickles/"+common
            path_labels_msp = "results/labels/"+common
            path_labels = "results/labels/"+common
            path_presegs = "results/presegs/"+common
            path_embeddings = "results/embeddings/"+common
            path_clusterings = "results/clusterings/"+common

            makedirs(path_graphs,exist_ok=True)
            makedirs(path_pickles,exist_ok=True)
            makedirs(path_labels,exist_ok=True)
            makedirs(path_presegs,exist_ok=True)
            makedirs(path_embeddings,exist_ok=True)
            makedirs(path_clusterings,exist_ok=True)
    
            pickle.dump(labels,open(path_labels+str(i+1)+"_"+filename[:-4]+".preseg","wb"))
            pickle.dump(segmentation,open(path_labels+str(i+1)+"_"+filename[:-4]+".seg","wb"))
            #pickle.dump(clusterings, open(path_clusterings+str(i+1)+"_"+filename[:-4]+".clt","wb"))
            numpy.save(path_embeddings+filename[:-4]+".emb",embeddings)
            nx.write_gpickle(Gr, path_pickles+str(i+1)+"_"+filename[:-4]+".pkl")
            nx.write_weighted_edgelist(Gr, path_graphs+filename[:-4]+".wgt", delimiter='\t')
            helper._savepreseg(labels, image, path_presegs+filename[:-4]+".png")
        else:
            from skimage.segmentation import mark_boundaries
            import matplotlib.pyplot as plt
            
            colored_regions = color.label2rgb(segmentation, image, alpha=1, colors=helper._colors(segmentation,image), bg_label=0)
            
            fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
            ax[0].imshow(mark_boundaries(image, labels))
            ax[0].set_title("initial segmentation")
            ax[1].imshow(mark_boundaries(colored_regions, segmentation, mode='thick'))
            ax[1].set_title('final segmentation')
            
            for a in ax.ravel():
                a.set_axis_off()
            
            plt.tight_layout()
            plt.show()
