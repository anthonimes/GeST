# import the necessary packages
from skimage.future import graph
from skimage import io, color

from os import walk, makedirs 

import networkx as nx
import numpy
import warnings
warnings.filterwarnings("ignore")

import pickle
import helper

import gest
#import pandas as pd

if __name__ == "__main__":
    
    # construct the argument parser and parse the arguments
    # common arguments
    # TODO: should maybe return the GeST instance?
    method, write, silhouette, merge, n_cluster, path_images, _sigma = helper._parse_args()

    # meanshift and SLIC arguments
    '''if method == "SLIC":
        _num_segments = float(argsy['segments'])
        _compactness = float(argsy['compactness'])
        common=method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"
    else:
        _spatial_radius=int(argsy['hs']) #hs
        _range_radius=float(argsy['hr']) #hr
        _min_density=int(argsy['mind']) #mind
        common=method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"'''
     
    common="MSP_7_4.5_50_125/"
    path_segmentation = "results/segmentation/"+common
    makedirs(path_segmentation,exist_ok=True)

    dirpath,_,images = list(walk(path_images))[0]

    for i,filename in enumerate(sorted(images)):
        # load the image and convert it to a floating point data type
        g = gest.GeST(dirpath+filename, n_cluster, preseg_method=method)
        g.segmentation()
        print(g.__dict__)
        
        # scaling features
        #helper._savefig(segmentation, image, path_segmentation+str(i+1)+"_"+filename[:-4]+"_"+str(n_cluster)+".png")

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
    
            pickle.dump(g._presegmentation,open(path_labels+str(i+1)+"_"+filename[:-4]+".preseg","wb"))
            pickle.dump(g._segmentation,open(path_labels+str(i+1)+"_"+filename[:-4]+".seg","wb"))
            #pickle.dump(clusterings, open(path_clusterings+str(i+1)+"_"+filename[:-4]+".clt","wb"))
            numpy.save(path_embeddings+filename[:-4]+".emb",g._embeddings)
            nx.write_gpickle(g._RAG, path_pickles+str(i+1)+"_"+filename[:-4]+".pkl")
            nx.write_weighted_edgelist(g._RAG, path_graphs+filename[:-4]+".wgt", delimiter='\t')
            helper._savepreseg(g._presegmentation, g._image, path_presegs+filename[:-4]+".png")
        else:
            from skimage.segmentation import mark_boundaries
            import matplotlib.pyplot as plt
            
            colored_regions = color.label2rgb(g._segmentation, g._image, alpha=1, colors=helper._colors(g._segmentation,g._image), bg_label=0)
            
            fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
            ax[0].imshow(mark_boundaries(g._image, g._presegmentation))
            ax[0].set_title("initial segmentation")
            ax[1].imshow(mark_boundaries(colored_regions, g._segmentation, mode='thick'))
            ax[1].set_title('final segmentation')
            
            for a in ax.ravel():
                a.set_axis_off()
            
            plt.tight_layout()
            plt.show()
