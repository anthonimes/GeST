# import the necessary packages
from skimage.segmentation import slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, img_as_ubyte
from skimage.future import graph
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from skimage.metrics import (adapted_rand_error,
                              variation_of_information,mean_squared_error)
from skimage import io,color,measure
from sklearn import cluster as cl

import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.parse_matlab import get_groundtruth, get_BSR
from utils.graph import _distance_r_graph, _get_Lab_adjacency, _get_histo_adjacency, _color_features
from utils import save
from utils.metrics.pri import probabilistic_rand_index
import utils.pyFmax.pyfmax.fmax as fm

from os import walk, environ
from statistics import mean, stdev
from math import sqrt, nan
import numpy
import networkx as nx
import warnings, sys, argparse
warnings.filterwarnings("ignore")

# https://github.com/fjean/pymeanshift
import pymeanshift as pms
# used by pymeanshift
import cv2,pickle,csv

# for reproducibility
SEED = 42
environ["PYTHONHASHSEED"] = str(SEED)

def metrics(im_true,im_test,verbose=False):
    error, precision, recall = adapted_rand_error(im_true, im_test)
    splits, merges = variation_of_information(im_true, im_test)
    if(verbose):
        print(f"Adapted Rand error: {error}")
        print(f"Adapted Rand precision: {precision}")
        print(f"Adapted Rand recall: {recall}")
        print(f"False Splits: {splits}")
        print(f"False Merges: {merges}")
    return error, precision, recall, splits, merges

def _groundtruth_clustering(labels,GT):
    regions = measure.regionprops(labels)
    clustering = numpy.zeros(len(regions))
    for i,region in enumerate(regions):
        max_dict = {c: 0 for c in range(1,numpy.amax(GT)+1)}
        for (x,y) in region.coords:
            max_dict[GT[x,y]]+=1
        cluster=int(max(max_dict.items(),key=lambda x: x[1])[0])
        clustering[i]=cluster# if cluster<=len(regions) else len(regions)
    return numpy.asarray(clustering)
    
def _best_groundtruth(GT):
    matrice=numpy.zeros((len(GT),len(GT)))
    positions = [(i,j) for i in range(len(GT)) for j in range(len(GT)) if i != j]
    for i,j in positions:
        # we compure the ARI between every pair of path_groundtruths
        ari,error,recall = adapted_rand_error(GT[i],GT[j])
        matrice[i,j]=ari
    # the best groundtruth is the one maximizing sum(row) or sum(column)
    return matrice.sum(axis=0).argmax()
    
def _scores(filename,path,G,Gr,feature_vector,feature_vector_pixels,NUM_CLUSTERS, embeddings,embeddings_pixels,image,labels,method="",groundtruth=None,GT=None,verbose=False,_dt=2,dataset="train",name=None,sigma=50):
    # TODO: write a file for EVERY groundtruth, to see if something better comes up
    with open(dataset+"_"+name+".csv", "a",newline='') as f:
        fieldnames = ["image","PRE","ARE","MSE","VOI","modregions","silemb","silfeatcolor","silfeathog","dbemb","dbfeat_color","db_feat_hog","chemb","chfeat_color","chfeat_hog","n_clusters","mean_n_clusters","stdev_n_clusters","min_weight_Gc","max_weight_Gc","mean_weight_Gc"]
        csvwriter = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        max_pri=0
        clusterings=pickle.load(open(path_clusterings+filename+".clt","rb"))

        # FIXME: CALCULER LE FEATURE VECTOR DU CLUSTERING ?
        for labels_clustering in clusterings:
            best_groundtruth = GT[_best_groundtruth(GT)]
            current_image_scores = [filename]
            #f.write(str(filename)+";")
            labels_from_clustering = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
            for i,line in enumerate(labels):
                for j,value in enumerate(line):
                    # +1 needed since 0=background
                    labels_from_clustering[i][j] = labels_clustering[value-1]+1
                        
            pri = probabilistic_rand_index(GT,labels_from_clustering)
            if(pri > max_pri):
                max_pri=pri
            # we need error to maybe consider minimum mean as best clustering
            current_image_scores.append(1-pri)
            ari,precision,recall = adapted_rand_error(best_groundtruth,labels_from_clustering)
            current_image_scores.extend([ari])
            
            mse = mean_squared_error(labels_from_clustering,best_groundtruth)
            voi = variation_of_information(labels_from_clustering,best_groundtruth)
            current_image_scores.extend([mse,voi[0]+voi[1]])
            # AT THIS STEP we have all errors: PRE, ARE, MSE, VOI
            '''partition = dict()
            labels_clustering_list = labels_from_clustering.flatten().tolist()
            for i,node in enumerate(G.nodes()):
                # assigning cluster as community for every node
                partition[node] =labels_clustering_list[node-1]
            #print("modularity on {} vertices and {} edges ({} communities)".format(len(G),len(G.edges()),numpy.amax(partition.values())))
            f.write(str(community_louvain.modularity(partition,G))+";")'''

            partition = dict()
            for i,node in enumerate(Gr.nodes()):
                # assigning cluster as community for every node
                partition[node] = labels_clustering[node-1]
            current_image_scores.append(community_louvain.modularity(partition,Gr))

            if(numpy.amax(labels_from_clustering)>1):
                silhouette_avg_embeddings = silhouette_score(embeddings, labels_clustering, metric='cosine')
                silhouette_avg_features_color = silhouette_score(feature_vector[:,:6], labels_clustering, metric='cosine')
                silhouette_avg_features_hog = silhouette_score(feature_vector[:,6:], labels_clustering, metric='cosine')
            
                davies_bouldin_embeddings = davies_bouldin_score(embeddings, labels_clustering)
                davies_bouldin_features_color = davies_bouldin_score(feature_vector[:,:6], labels_clustering)
                davies_bouldin_features_hog = davies_bouldin_score(feature_vector[:,6:], labels_clustering)

                calinski_harabasz_embeddings = calinski_harabasz_score(embeddings,labels_clustering) 
                calinski_harabasz_features_color = calinski_harabasz_score(feature_vector[:,:6],labels_clustering) 
                calinski_harabasz_features_hog = calinski_harabasz_score(feature_vector[:,6:],labels_clustering)
            
                Gc = graph.rag_mean_color(image,labels_from_clustering,connectivity=2,mode='similarity',sigma=_sigma)
                
                weights = [e[2] for e in Gc.edges(data='weight')]
                min_Gc,max_Gc,mean_Gc = min(weights), max(weights), mean(weights)
            else:
                silhouette_avg_embeddings = nan
                silhouette_avg_features_color = nan
                silhouette_avg_features_hog = nan
                
                davies_bouldin_embeddings = nan
                davies_bouldin_features_color = nan
                davies_bouldin_features_hog = nan
                
                calinski_harabasz_embeddings = nan
                calinski_harabasz_features_color = nan
                calinski_harabasz_features_hog = nan
                
                min_Gc, max_Gc, mean_Gc = nan, nan, nan
                
            current_image_scores.extend([silhouette_avg_embeddings,silhouette_avg_features_color,silhouette_avg_features_hog])
            
            current_image_scores.extend([davies_bouldin_embeddings,davies_bouldin_features_color,davies_bouldin_features_hog])
            
            current_image_scores.extend([calinski_harabasz_embeddings,calinski_harabasz_features_color,calinski_harabasz_features_hog])

            current_image_scores.append(numpy.amax(labels))
            current_image_scores.append(mean(numpy.bincount(labels[labels>=1].flatten())))
            current_image_scores.append(stdev(numpy.bincount(labels[labels>=1].flatten())))
            
            current_image_scores.extend([min_Gc,max_Gc,mean_Gc]) 
            
            # computing weights of the RAG associated to the clustering: maybe useless?
            
            '''obj_embeddings=fm.MatrixClustered(embeddings, labels_clustering)
            try:
                f.write(str(obj_embeddings.get_macro_PC())+";")
            except:
                f.write("0;")
            try:
                f.write(str(obj_embeddings.get_avg_PC())+";")
            except:
                f.write("0;")
            try:
                f.write(str(obj_embeddings.get_macro_EC())+";")
            except:
                f.write("0;")
            try:
                f.write(str(obj_embeddings.get_avg_EC())+";")
            except:
                f.write("0;")

            obj_features=fm.MatrixClustered(feature_vector, labels_clustering)
            try:
                f.write(str(obj_features.get_macro_PC())+";")
            except:
                f.write("0;")
            try:
                f.write(str(obj_features.get_avg_PC())+";")
            except:
                f.write("0;")
            try:
                f.write(str(obj_features.get_macro_EC())+";")
            except:
                f.write("0;")
            try:
                f.write(str(obj_features.get_avg_EC())+"\n")
            except:
                f.write("0\n")'''
            csvwriter.writerow({fieldnames[i]: current_image_scores[i] for i in range(len(current_image_scores))})

        # dealing with groundtruth at last --- ONLY FOR TRAIN AND TEST SETS
        '''if(dataset != "val"):
            for pos,gt in enumerate(GT):
                groundtruth = _groundtruth_clustering(labels,gt).astype(int)
                if(len(numpy.unique(groundtruth))<numpy.amax(labels)) and NUM_CLUSTERS != [] and len(numpy.unique(groundtruth))>1:
                    f.write(str(filename+".GT")+";")

                    labels_from_clustering = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
                    for i,line in enumerate(labels):
                        for j,value in enumerate(line):
                            # +1 needed since 0=background
                            labels_from_clustering[i][j] = groundtruth[value-1]
                                
                    pri = probabilistic_rand_index(GT,labels_from_clustering)
                    f.write(str(pri)+";")

                    labels_clustering_list = labels_from_clustering.flatten().tolist()
                    partition = dict()
                    for i,node in enumerate(G.nodes()):
                        # assigning cluster as community for every node
                        partition[node] = labels_clustering_list[node-1]
                    f.write(str(community_louvain.modularity(partition,G))+";")

                    partition = dict()
                    for i,node in enumerate(Gr.nodes()):
                        # assigning cluster as community for every node
                        partition[node] = groundtruth[node-1]
                    f.write(str(community_louvain.modularity(partition,Gr))+";")

                    silhouette_avg_embeddings = silhouette_score(embeddings, groundtruth, metric='euclidean')
                    silhouette_avg_features_color = silhouette_score(feature_vector[:,:6], groundtruth, metric='euclidean')
                    silhouette_avg_features_hog = silhouette_score(feature_vector[:,6:], groundtruth, metric='euclidean')
                    silhouette_avg_features = silhouette_score(feature_vector, groundtruth, metric='euclidean')
                    f.write(str(silhouette_avg_embeddings)+";"+str(silhouette_avg_features_color)+";"+str(silhouette_avg_features_hog)+";")

                    davies_bouldin_embeddings = davies_bouldin_score(embeddings, groundtruth)
                    davies_bouldin_features_color = davies_bouldin_score(feature_vector[:,:6], groundtruth)
                    davies_bouldin_features_hog = davies_bouldin_score(feature_vector[:,6:], groundtruth)
                    f.write(str(davies_bouldin_embeddings)+";"+str(davies_bouldin_features_color)+";"+str(davies_bouldin_features_hog)+";")

                    calinski_harabasz_embeddings = calinski_harabasz_score(embeddings,groundtruth) 
                    calinski_harabasz_features_color = calinski_harabasz_score(feature_vector[:,:6],groundtruth) 
                    calinski_harabasz_features_hog = calinski_harabasz_score(feature_vector[:,6:],groundtruth) 
                    f.write(str(calinski_harabasz_embeddings)+";"+str(calinski_harabasz_features_color)+";"+str(calinski_harabasz_features_hog)+";")
                    
                    f.write(str(numpy.amax(groundtruth))+";")
                    f.write(str(mean(numpy.bincount(groundtruth)))+";")
                    f.write(str(stdev(numpy.bincount(groundtruth)))+"\n")

                else:
                    f.write("\n")'''

        print(1-max_pri)

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = False, help = "Path to the image")
    ap.add_argument("-p", "--path", required = False, help = "Path to folder")
    ap.add_argument("-m", "--method", required = True, help="pre-segmentation method")
    ap.add_argument("-b", "--best", required = False, help="compute best clustering?", default=False)
    ap.add_argument("-w", "--write", required = False, help="write all files to hard drive?", default=False)
    ap.add_argument("-d", "--dataset", required = False, help="which of {train,val,test} to evaluate?", default="val")
    ap.add_argument("--hs", required = False, help="spatial radius?", default=15)
    ap.add_argument("--hr", required = False, help="range radius?", default=4.5)
    ap.add_argument( "--mind", required = False, help="min density", default=300)
    ap.add_argument( "--sigma", required = True, help="kernel parameter", default=50)
    ap.add_argument( "--segments", required = False, help="number of segments (SLIC)", default=50)
    ap.add_argument( "--compactness", required = False, help="compactness (SLIC)", default=50)

    argsy = vars(ap.parse_args())
    path_image = argsy['path']+"/images/"
    path_groundtruth = argsy['path']+"/groundTruth/"

    path_results = "results/"
    _spatial_radius=int(argsy['hs']) #hs
    _range_radius=float(argsy['hr']) #hr
    _min_density=int(argsy['mind'])
    _sigma=float(argsy['sigma'])
    _num_segments = float(argsy['segments'])
    _compactness = float(argsy['compactness'])

    methods = { "slic": "SLIC", "msp": "MSP", "mso": "MSO" }
    method = methods[argsy['method']]

    which_folder = {"val": "val/", "train": "train/", "test": "test/"}
    folder = which_folder[argsy['dataset']]

    path_groundtruths = path_groundtruth+folder

    if method == "SLIC":
        path_pickles = "results/pickles/"+method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_labels = "results/labels/"+method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_embeddings = "results/embeddings/"+method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_clusterings = "results/clusterings/"+method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"+folder
        name = method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)
    else:
        path_pickles = "results/pickles/"+method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_labels = "results/labels/"+method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_embeddings = "results/embeddings/"+method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_clusterings = "results/clusterings/"+method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder
        name = method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)

    # computing best clustering ?
    argsy['best'] = True if argsy['best'] == "True" else False
    if argsy['best'] is True:
        path_pickles+="best/"
        path_labels+="best/"
        path_embeddings+="best/"
        path_clusterings+="best/"

    # load the image and convert it to a floating point data type
    for (dirpath, dirnames, filenames) in walk(path_pickles):
        for i,filename in enumerate(sorted(filenames)):
            if filename.endswith(".pkl"):
                print("{}: {}".format(i+1,filename[:-4]),end=' ')
                image = io.imread(path_image+folder+filename[:-4]+".jpg")
                image_lab = color.rgb2lab(image)
                image_lab = (color.rgb2lab(image) + [0,128,128]) #// [1,1,1]
                gt_boundaries, gt_segmentation = get_groundtruth(path_groundtruths+filename[:-4]+".mat")
                
                Gr = nx.read_gpickle(path_pickles+filename[:-4]+".pkl")
                # creating image graph
                width = image.shape[1]
                height = image.shape[0]
                
                labels = save._loadlabels(path_labels+filename[:-4]+".preseg")
                number_regions = numpy.amax(labels)
                
                NUM_CLUSTERS = list(range(1,min(26,number_regions)))
                
                embn2v = save._loadembeddings(path_embeddings+filename[:-4]+".emb.npy")
                feature_vector = _color_features(labels,image_lab)

                # deducing embeddings for the whole image
                embeddings = [0]*(width*height)
                FV = [0]*(width*height)
                '''for i,region in enumerate(measure.regionprops(labels)):
                    for (x,y) in region.coords:
                        # starts from 0 right?
                        embeddings[(x*width)+y]=embn2v[i]

                # deducing feature_vector for the whole image
                for i,region in enumerate(measure.regionprops(labels)):
                    for (x,y) in region.coords:
                        # starts from 0 right?
                        FV[(x*width)+y]=feature_vector[i]'''

                #best_k = elbow(embn2v,min(25,number_regions))
                #best_k = silhouette(numpy.asarray(embn2v),min(25,number_regions))
                _scores(filename[:-4],path_clusterings,None,Gr,numpy.asarray(feature_vector),numpy.asarray(FV),NUM_CLUSTERS,numpy.asarray(embn2v),numpy.asarray(embeddings),image_lab,numpy.asarray(labels),method=method,groundtruth=gt_segmentation[0], GT=gt_segmentation,_dt=1.2,dataset=argsy['dataset'],name=name,sigma=_sigma)
                #_scores_vs_groundtruth(filename[:-4],path_clusterings,None,Gr,numpy.asarray(feature_vector),numpy.asarray(FV),NUM_CLUSTERS,numpy.asarray(embn2v),numpy.asarray(embeddings),image_lab,numpy.asarray(labels),method=method,groundtruth=gt_segmentation[0], GT=gt_segmentation,_dt=1.2,dataset=argsy['dataset'],name=name,sigma=_sigma)
                
