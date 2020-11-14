# import the necessary packages
from skimage.segmentation import slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, img_as_ubyte
from skimage.future import graph
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from skimage.metrics import (adapted_rand_error,
                              variation_of_information)
from skimage import io,color,measure
from sklearn import cluster as cl
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import scale
from pyclustertend import hopkins, vat, ivat

from sklearn.preprocessing import StandardScaler

import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.parse_matlab import get_groundtruth, get_BSR
from utils.graph import _distance_r_graph, _get_Lab_adjacency, _get_histo_adjacency, _color_features
from utils.save import _savefig, _savelabels, _savelabels_seg, _loadlabels, _loadembeddings
from utils.metrics.pri import probabilistic_rand_index
import utils.pyFmax.pyfmax.fmax as fm

from node2vec.model import Node2Vec

from sklearn.decomposition import PCA

from os import walk, environ
from statistics import mean, stdev
from math import sqrt
import numpy
import networkx as nx
import warnings, sys, argparse
warnings.filterwarnings("ignore")

# https://github.com/fjean/pymeanshift
import pymeanshift as pms
# used by pymeanshift
import cv2

# for reproducibility
SEED = 19
environ["PYTHONHASHSEED"] = str(SEED)

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = False, help = "Path to the image")
    ap.add_argument("-p", "--path", required = False, help = "Path to folder")
    ap.add_argument("-m", "--method", required = True, help="pre-segmentation method")
    ap.add_argument("-b", "--best", required = False, help="compute best clustering?", default=False)
    ap.add_argument("-w", "--write", required = False, help="write all files to hard drive?", default=False)
    ap.add_argument("-d", "--dataset", required = False, help="which of {train,val,test} to evaluate?", default="val")

    argsy = vars(ap.parse_args())
    path_image = argsy['path']+"/images/"
    path_groundtruth = argsy['path']+"/groundTruth/"

    path_results = "results/"
    _spatial_radius=15
    _range_radius=4.5
    _min_density=300
    _sigma=50
    _num_segments = 200
    _compactness = 75

    methods = { "slic": "SLIC", "msp": "MSP", "mso": "MSO" }
    method = methods[argsy['method']]

    which_folder = {"val": "val/", "train": "train/", "test": "test/"}
    folder = which_folder[argsy['dataset']]

    path_groundtruths = path_groundtruth+folder

    if method == "slic":
        path_pickles = "results/pickles/"+method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_labels = "results/labels/"+method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_embeddings = "results/embeddings/"+method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"+folder
    else:
        path_pickles = "results/pickles/"+method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_labels = "results/labels/"+method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_embeddings = "results/embeddings/"+method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder

    # computing best clustering ?
    argsy['best'] = True if argsy['best'] == "True" else False
    if argsy['best'] is True:
        path_pickles+="best/"
        path_labels+="best/"
        path_embeddings+="best/"

    FELZ_PRI, BSR_PRI, LOUVAIN_PRI, GEST_PRI, GEST_EMB_PRI, EMB_PRI, BSL_PRI, LLE_PRI, HOPE_PRI, HOPE_EMB_PRI = [], [], [], [], [], [], [], [], [], []
    # load the image and convert it to a floating point data type
    for (dirpath, dirnames, filenames) in walk(path_pickles):
        for i,filename in enumerate(filenames):
            if filename.endswith(".pkl"):
                print("{}: {}".format(i+1,filename[:-4]),end=' ')
                image = io.imread(path_image+folder+filename[:-4]+".jpg")
                image_lab = color.rgb2lab(image)
                image_lab = (color.rgb2lab(image) + [0,128,128]) #// [1,1,1]
                gt_boundaries, gt_segmentation = get_groundtruth(path_groundtruths+filename[:-4]+".mat")
                # labels with 0 are ignored, seems legit? --- ENFORCE STARTING AT 1, START_LABEL IS FUCKING DUMP
                #labels = 1+slic(image, n_segments = numSegments, compactness=75, convert2lab=True, start_label=0)
                #number_regions = numpy.amax(labels)

                # FIXME: LOAD GRAPH WITH NETWORKX FOR BETTER COMPUTING TIME
                #(segmented_image, labels, number_regions) = pms.segment(ms_image, spatial_radius=15, range_radius=4.5, min_density=300)
                #labels = 1+labels
                #labels = 1+quickshift(image, kernel_size=2, max_dist=6, ratio=0.5)
                #print("number of regions from MeanShift: {}".format(number_regions))

                '''_savefig(labels,image,path_results+"SLIC/"+filename[:-4]+".png")            

                for i in range(len(gt_segmentation)):
                    _savefig(gt_segmentation[i], image, path_results+"/groundtruth/images/"+filename[:-4]+"_"+str(i+1)+".png",colored=True)
                    _savelabels(gt_segmentation[i],path_results+"/groundtruth/labels/"+filename[:-4]+"_"+str(i+1)+".lbl")
                    _savelabels_seg(gt_segmentation[i],path_results+"/groundtruth/segmentation/"+filename[:-4]+"_"+str(i+1)+".seg",filename)'''

                embn2v = _loadembeddings(path_embeddings+filename[:-4]+".emb.npy")
                X=scale(embn2v)
                print(hopkins(X,embn2v.shape[0]))
                #mat=vat(X)
                imat=ivat(X)
                #plt.show()

