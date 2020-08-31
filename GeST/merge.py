# import the necessary packages
from skimage.util import img_as_float, img_as_ubyte
from skimage.future import graph
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.spatial import distance

from skimage import io,color,measure

from utils.parse_matlab import get_groundtruth, get_BSR
from utils.graph import _color_features, _get_Lab_adjacency
from utils import save
from utils.metrics.pri import probabilistic_rand_index

from os import walk, environ, makedirs
from statistics import mean, stdev
import numpy, helper
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

def merge(labels,image_lab,thr_pixels=200,thr=0.995,sigma=5):
    # NOTE; labels must be a matrix-like imaeg
    labels_merge = numpy.copy(labels)
    merged=True
    has_merged=False
    # initial computation, will be maintained during algorithm
    feature_vector = normalize(numpy.asarray(_color_features(labels_merge,image_lab)))
    G = graph.RAG(labels_merge,connectivity=2)
    while(merged):
        regions = measure.regionprops(labels_merge)
        # FIXME: totally useless to compute again the ones that have not changed...
        merged=False
        
        def _findregion(R):
            for i in range(len(regions)):
                if regions[i].label == R:
                    return i
        
        for u,v in G.edges():
            Ri=regions[_findregion(u)]
            Rj=regions[_findregion(v)]
            sim=1-distance.cosine(feature_vector[Ri.label-1],feature_vector[Rj.label-1])
            if sim >= thr:
                #print("similarity merging region {} and {}.".format(Ri.label,Rj.label))
                max_label = Ri if Ri.label > Rj.label else Rj
                min_label = Ri if Ri.label < Rj.label else Rj
                for (x,y) in max_label.coords:
                    labels_merge[(x,y)] = min_label.label
                merged=True
                has_merged=True
                # updating remaining labels
                #_, labels_merge = numpy.unique(labels_merge,return_inverse=1)
                #labels_merge=(1+labels_merge).reshape(labels.shape)
                # updating feature vector
                feature_vector[min_label.label-1] = (feature_vector[min_label.label-1]+feature_vector[max_label.label-1])/2
                G = nx.contracted_nodes(G,min_label.label,max_label.label,self_loops=False)
                #print("COSI",(feature_vector[min_label.label-1]+feature_vector[max_label.label-1])/2)
            if(merged):
                break
        if(merged):
            continue
                
        # trying to merge small regions to their most similar neighbors
        # FIXME: IS IT BETTER AFTER OR BEFORE MERGING SMALL REGIONS?
        for i in range(len(regions)):
            Ri = regions[i]
            lenRi = len(Ri.coords)
            if(lenRi < thr_pixels):
                # WARNING: neighbors in graphs are labels, not indices of regions array!
                neighbors = list(G.neighbors(Ri.label))
                closest = max([(regions[_findregion(Rj)].label,1-distance.cosine(feature_vector[Ri.label-1],feature_vector[regions[_findregion(Rj)].label-1])) for Rj in neighbors],key=lambda x: x[1])[0]
                Rj = regions[_findregion(closest)]
                max_label = Ri if Ri.label > Rj.label else Rj
                min_label = Ri if Ri.label < Rj.label else Rj
                # could this actually be enough?
                #max_label.label = min_label.label
                for (x,y) in max_label.coords:
                    labels_merge[(x,y)] = min_label.label
                merged=True
                has_merged=True
                # updating remaining labels
                #_, labels_merge = numpy.unique(labels_merge,return_inverse=1)
                #labels_merge=(1+labels_merge).reshape(labels.shape)
                # updating feature vector
                #print("PIXE",(feature_vector[min_label.label-1]+feature_vector[max_label.label-1])/2)
                feature_vector[min_label.label-1] = (feature_vector[min_label.label-1]+feature_vector[max_label.label-1])/2
                G = nx.contracted_nodes(G,min_label.label,max_label.label,self_loops=False)
            if(merged):
                break
        if(merged):
            continue
        
    _, labels_merge = numpy.unique(labels_merge,return_inverse=1)
    labels_merge=(1+labels_merge).reshape(labels.shape)
    return labels_merge,has_merged

# FIXME: try again to merge ALL images with small pixels, just to see improvements
# FIXME: WORK ON CLUSTERING YOU FUCKING IDIOT
def merge_pixels(clusters,labels,image_lab,thr_pixels=300,sigma=5):
    # NOTE; labels must be a matrix-like image
    clusters_merge = numpy.copy(clusters)
    labels_merge = numpy.copy(labels)
    merged=True
    has_merged=False
    # initial computation, will be maintained during algorithm
    feature_vector = normalize(numpy.asarray(helper._color_features(labels,image_lab)))
    G = graph.RAG(labels,connectivity=2)
    while(merged):
        #regions = measure.regionprops(labels_merge)
        # FIXME: totally useless to compute again the ones that have not changed...
        merged=False
                
        for i in numpy.unique(clusters):
            Ri = i+1
            # how many pixels have this label?
            lenRi = len(numpy.where(labels_merge.flatten()==Ri)[0])
            if(lenRi > 0 and lenRi < thr_pixels and Ri in G.nodes()):
                # WARNING: neighbors in graphs are labels, not indices of regions array!
                neighbors = list(G.neighbors(Ri))
                if(neighbors):
                    closest = max([(Rj,1-distance.cosine(feature_vector[Ri-1],feature_vector[Rj-1])) for Rj in neighbors],key=lambda x: x[1])[0]
                    Rj = closest
                    print("merging regions...",Ri,Rj)
                    sim=1-distance.cosine(feature_vector[Ri-1],feature_vector[Rj-1])
                    #if(sim>=0.996):
                    max_label = Ri if Ri > Rj else Rj
                    min_label = Ri if Ri < Rj else Rj
                    labels_merge[numpy.where(labels_merge==max_label)] = min_label
                    clusters_merge[numpy.where(clusters_merge==max_label-1)] = min_label-1
                    merged=True
                    has_merged=True
                    # updating remaining labels
                    #_, labels_merge = numpy.unique(labels_merge,return_inverse=1)
                    #labels_merge=(1+labels_merge).reshape(labels.shape)
                    # updating feature vector
                    #print("PIXE",(feature_vector[min_label.label-1]+feature_vector[max_label.label-1])/2)
                    feature_vector[min_label-1] = (feature_vector[min_label-1]+feature_vector[max_label-1])/2
                    G = nx.contracted_nodes(G,min_label,max_label,self_loops=False)
            if(merged):
                break
        if(merged):
            continue
    
    if(has_merged):
        _, clusters_merge = numpy.unique(clusters_merge,return_inverse=1)
        clusters_merge=(1+clusters_merge).reshape(clusters.shape)
    return clusters_merge,has_merged

# FIXME: try again to merge ALL images with small pixels, just to see improvements
def merge_cosine(clusters,labels,image_lab,thr=0.999,sigma=5):
    # NOTE; labels must be a matrix-like image
    clusters_merge = numpy.copy(clusters)
    #labels_merge = numpy.copy(labels)
    merged=True
    has_merged=False
    # initial computation, will be maintained during algorithm
    feature_vector = normalize(numpy.asarray(helper._color_features(labels,image_lab)))
    G = graph.RAG(labels,connectivity=2)
    while(merged):
        # FIXME: totally useless to compute again the ones that have not changed...
        merged=False
        
        for u,v in G.edges():
            Ri=u
            Rj=v
            sim=1-distance.cosine(feature_vector[Ri-1],feature_vector[Rj-1])
            #sim=1-distance.euclidean(feature_vector[Ri.label-1],feature_vector[Rj.label-1])
            if sim >= thr:
                #print("similarity merging region {} and {}.".format(Ri.label,Rj.label))
                max_label = Ri if Ri > Rj else Rj
                min_label = Ri if Ri < Rj else Rj
                clusters_merge[numpy.where(clusters_merge==max_label-1)] = min_label-1
                merged=True
                has_merged=True
                # updating remaining labels
                #_, labels_merge = numpy.unique(labels_merge,return_inverse=1)
                #labels_merge=(1+labels_merge).reshape(labels.shape)
                # updating feature vector
                feature_vector[min_label-1] = (feature_vector[min_label-1]+feature_vector[max_label-1])/2
                G = nx.contracted_nodes(G,min_label,max_label,self_loops=False)
                #print("COSI",(feature_vector[min_label.label-1]+feature_vector[max_label.label-1])/2)
            if(merged):
                break
        if(merged):
            continue
        
    _, clusters_merge = numpy.unique(clusters_merge,return_inverse=1)
    clusters_merge=(1+clusters_merge).reshape(clusters.shape)
    return clusters_merge,has_merged

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

    which_folder = {"val": "val/", "train": "train/", "test": "test/", "hard_msp": "hard_msp/", "impossible": "impossible/"}
    folder = which_folder[argsy['dataset']]

    path_groundtruths = path_groundtruth+folder
    path_images = argsy['path']+"/images/"+folder
    #path_impossible = argsy['path']+"/images/from_BSR"

    if method == "SLIC":
        common=method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_graphs = "results/graphs/"+common
        path_pickles = "results/pickles/"+common
        path_labels = "results/labels/"+common
        path_scores = "results/scores/"+common
        path_figs = "results/figs/"+common
        path_presegs = "results/presegs/"+common
        path_embeddings = "results/embeddings/"+common
        path_clusterings = "results/clusterings/"+common
        path_matlab = "results/matlab/"+common
    else:
        common=method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_graphs = "results/graphs/"+common
        path_pickles = "results/pickles/"+common
        path_labels = "results/labels/"+common
        path_scores = "results/scores/"+common
        path_figs = "results/figs/"+common
        path_presegs = "results/presegs/"+common
        path_embeddings = "results/embeddings/"+common
        path_clusterings = "results/clusterings/"+common
        path_matlab = "results/matlab/"+common

    if argsy['best'] == "True":
        path_graphs+="best/"
        path_pickles+="best/"
        path_labels+="best/"
        path_figs+="best/"
        path_presegs+="best/"
        path_embeddings+="best/"
        path_clusterings+="best/"
        path_matlab+="best/"

    makedirs(path_figs,exist_ok=True)

    MERGED, RENUM, NORMAL = [], [], []
    path_impossible = argsy['path']+"/images/from_observation"
    dirpath, dirnames, hardimages = list(walk(path_impossible))[0]
    
    # load the image and convert it to a floating point data type
    for (dirpath, dirnames, filenames) in walk(path_images):
        for i,filename in enumerate(sorted(filenames)):
            if filename.endswith(".jpg"):
                #print("{}: {}".format(i+1,filename[:-4]),end=' ')
                image = io.imread(path_image+"val/"+filename[:-4]+".jpg")
                image = img_as_float(image)
                image_lab = color.rgb2lab(image)
                image_lab = (color.rgb2lab(image) + [0,128,128]) #// [1,1,1]
                gt_boundaries, gt_segmentation = get_groundtruth(path_groundtruths+filename[:-4]+".mat")
                
                Gr = nx.read_gpickle(path_pickles+str(i+1)+"_"+filename[:-4]+".pkl")
                # creating image graph
                labels = pickle.load(open(path_labels+str(i+1)+"_"+filename[:-4]+".seg","rb"))
                labels_preseg = pickle.load(open(path_labels+str(i+1)+"_"+filename[:-4]+".preseg","rb"))
                labels_clustering = pickle.load(open(path_clusterings+str(i+1)+"_"+filename[:-4]+".clt","rb"))[0].labels_
                number_regions = numpy.amax(labels)
                
                pri=probabilistic_rand_index(gt_segmentation,labels)
                
                # FIXME: FIND BEST AMONG CURRENT PARAMETERS WITH MERGE
                #if(filename in hardimages):
                #if(True):
                # MERGING CONTIGUOUS REGIONS ONLY IN A FIRST PLACE
                new_labels_clustering = numpy.copy(labels_clustering)
                segmentation = numpy.copy(labels)
                for _label in numpy.unique(labels_clustering):
                    labelmax = numpy.amax(new_labels_clustering)
                    # getting regions with this label
                    vertices = 1+numpy.argwhere(labels_clustering == _label).flatten()
                    # ugly but if connected, the if will fail...
                    Gc = Gr if len(Gr.subgraph(vertices).edges())==0 else Gr.subgraph(vertices)
                    connected_component = sorted(nx.connected_components(Gc), key=len, reverse=True)
                    if(len(connected_component)>1):
                        to_relabel=connected_component[1:]
                        labelcpt=1
                        for cc in to_relabel:
                            for vertex in cc:
                                new_labels_clustering[vertex-1]=labelmax+labelcpt
                            #print(numpy.unique(new_labels_clustering))
                            labelcpt+=1
                           
                segmentation = numpy.zeros(labels.shape,dtype=int)
                # computing corresponding new segmentation
                for l,line in enumerate(labels_preseg):
                    for j,value in enumerate(line):
                        segmentation[l][j] = new_labels_clustering[value-1]+1
                        
                helper._savefig(segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+"_RENUM.png") 
                            
                # contiguous labels
                #_, segmentation = numpy.unique(segmentation,return_inverse=1)
                #segmentation=(1+segmentation).reshape(labels.shape)
                        
                prirenum=probabilistic_rand_index(gt_segmentation,segmentation)
                NORMAL.append(pri)
                RENUM.append(prirenum)
                      
                if(filename in hardimages):
                    print("{}: {}".format(i+1,filename[:-4]))
                    #print("loading done: segmentation with {} labels and PRI {}.".format(numpy.amax(labels),probabilistic_rand_index(gt_segmentation,labels)))
                    #feature_vector = normalize(numpy.asarray(_color_features(labels,image_lab)))
                    #print("feature vector obtained.")
                    #thresholds=cosine_similarity(feature_vector[:,:6])
                    #_,thresholds=_get_Lab_adjacency(labels,image_lab,_sigma)
                    #print("similary matrix obtained")
                    print("after cosine...", probabilistic_rand_index(gt_segmentation,segmentation))
                            
                    #print(has_merged, end=' ')
                    # FIXME: MERGE ONLY IF A GIVEN THRESHOLD IS RESPECTED
                    final_clustering,has_merged=merge_pixels(new_labels_clustering,segmentation,image_lab,thr_pixels=200,sigma=_sigma)
                    
                    segmentation = numpy.zeros(labels.shape,dtype=int)
                    # computing corresponding new segmentation
                    for l,line in enumerate(labels_preseg):
                        for j,value in enumerate(line):
                            segmentation[l][j] = final_clustering[value-1]
                            
                    print("after both...",probabilistic_rand_index(gt_segmentation,segmentation))
                    
                    #if(filename in hardimages):
                    final_clustering,has_merged=merge_cosine(final_clustering,segmentation,image_lab,thr=0.9995,sigma=_sigma)
                    
                    segmentation = numpy.zeros(labels.shape,dtype=int)
                    # computing corresponding new segmentation
                    for l,line in enumerate(labels_preseg):
                        for j,value in enumerate(line):
                            segmentation[l][j] = final_clustering[value-1]
                            
                            
                    #print(has_merged)
                    #labels_merged, has_merged = merge(labels,image_lab,thr_pixels=100,thr=0.999,sigma=_sigma)
                    #print(numpy.amax(labels_merged),numpy.amin(labels_merged),len(numpy.unique(labels_merged)))
                    #labels_merged=segmentation
                    primerged=probabilistic_rand_index(gt_segmentation,segmentation)
                    print(numpy.amax(labels_clustering), len(numpy.unique(new_labels_clustering)), len(numpy.unique(final_clustering)),pri,prirenum,primerged)
                    #save._savefig(labels_merged, image, path_figs+str(i+1)+"_"+filename[:-4]+"_MERGED.png") 
                    #if(primerged>pri):
                    #if(True):
                    #    print("YOUPI! "+filename[:-4]+" merged: new segmentation has {} regions and PRI {} from {}".format(len(numpy.unique(labels_merged)),primerged,pri))
                    MERGED.append(primerged)
                    #helper._savefig(labels, image, path_figs+str(i+1)+"_"+filename[:-4]+".png") 
                    helper._savefig(segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+"_MERGED.png") 
                    #else:
                    #    BEST.append(pri)
                else:
                    MERGED.append(pri)
                #else:
                    #BEST.append(probabilistic_rand_index(gt_segmentation,labels))
    print("MEAN MERGED",mean(MERGED))
    print("MEAN NORMAL",mean(NORMAL))
    print("MEAN RENUM",mean(RENUM))
               
