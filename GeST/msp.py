# import the necessary packages
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.future import graph
from skimage import io, color, measure
from skimage.metrics import variation_of_information
from scipy.spatial import distance

from sklearn import cluster as cl
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score

from os import makedirs, environ, walk
from statistics import mean, stdev
from utils.graph import _color_features, _get_Lab_adjacency

# https://github.com/thibaudmartinez/node2vec
from node2vec.model import Node2Vec
from utils.node2vec.src import node2vec
#from fastnode2vec import Graph, Node2Vec
from gensim.models import Word2Vec

from merge import merge_pixels, merge_cosine

import numpy, csv
import networkx as nx
import warnings, sys, argparse
warnings.filterwarnings("ignore")

# used by pymeanshift
import pickle
import helper

'''args=Namespace()
args.P=1.0
args.Q=0.2
args.dimensions=128
args.window_size=5
args.walk_length=40
args.walk_number=20
args.input=None
args.output=None
args.walk_type="second"
args.workers=4
args.min_count=1

Gr.add_node(0)
max_pri=0
max_nb_clusters=0
wm = WalkletMachine(args,Gr)'''

# for reproducibility
SEED = 42
environ["PYTHONHASHSEED"] = str(SEED)

def merge_pixels(labels,image_lab,thr_pixels=300,sigma=5):
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
                
        for i in range(len(regions)):
            Ri = regions[i]
            lenRi = len(Ri.coords)
            if(lenRi < thr_pixels):
                # WARNING: neighbors in graphs are labels, not indices of regions array!
                neighbors = list(G.neighbors(Ri.label))
                closest = max([(regions[_findregion(Rj)].label,1-distance.cosine(feature_vector[Ri.label-1],feature_vector[regions[_findregion(Rj)].label-1])) for Rj in neighbors],key=lambda x: x[1])[0]
                Rj = regions[_findregion(closest)]
                sim=1-distance.cosine(feature_vector[Ri.label-1],feature_vector[Rj.label-1])
                if(sim>=0.996):
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
def merge_cosine(labels,image_lab,thr=0.999,sigma=5):
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
            #sim=1-distance.euclidean(feature_vector[Ri.label-1],feature_vector[Rj.label-1])
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
        
    _, labels_merge = numpy.unique(labels_merge,return_inverse=1)
    labels_merge=(1+labels_merge).reshape(labels.shape)
    return labels_merge,has_merged

def learn_embeddings(walks,dimensions=32,window_size=5,min_count=0,workers=4,iter=1):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=min_count, sg=1, workers=workers, iter=iter)
    #model.save_word2vec_format(args.output)
    
    return model

if __name__ == "__main__":
    methods = { "slic": "SLIC", "msp": "MSP", "mso": "MSO" }
    which_folder = {"val": "val/", "train": "train/", "test": "test/", "bug": "bug/"}
    
    # construct the argument parser and parse the arguments
    argsy = helper._parse_args()

    # meanshift and SLIC arguments
    _spatial_radius=int(argsy['hs']) #hs
    _range_radius=float(argsy['hr']) #hr
    _min_density=int(argsy['mind']) #mind
    _sigma=float(argsy['sigma'])
    _num_segments = float(argsy['segments'])
    _compactness = float(argsy['compactness'])
    
    method = methods[argsy['method']]
    
    # computing best clustering ?
    best = True if argsy['best'] == "True" else False
    write = True if argsy['write'] == "True" else False
    folder = which_folder[argsy['dataset']]
    
    path_images = argsy['path']+"/images/"+folder
    path_groundtruths = argsy['path']+"/groundTruth/"+folder
    
    if method == "SLIC":
        common=method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_csv= "results/best_scores_clusterings_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
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

    makedirs(path_graphs,exist_ok=True)
    makedirs(path_pickles,exist_ok=True)
    makedirs(path_labels,exist_ok=True)
    makedirs(path_scores,exist_ok=True)
    makedirs(path_figs,exist_ok=True)
    makedirs(path_presegs,exist_ok=True)
    makedirs(path_embeddings,exist_ok=True)
    makedirs(path_clusterings,exist_ok=True)
    makedirs(path_matlab,exist_ok=True)
    
    path_impossible = argsy['path']+"/images/hard_msp"
    dirpath, dirnames, hardimages = list(walk(path_impossible))[0]

    BEST_GEST_PRI, AVG_GEST_PRI, MSP_PRI, EMB_PRI, BOTH_PRI, VOI, MIS, VOIMSP, MISMSP = [], [], [], [], [], [], [], [], []
    for (dirpath, dirnames, filenames) in walk(path_images):
        for i,filename in enumerate(sorted(filenames)):
            if filename.endswith(".jpg"):
                print("{}: {}".format(i+1,filename))
                # load the image and convert it to a floating point data type

                #image_lab = image
                image = io.imread(dirpath+filename)
                image = img_as_float(image)
                image_lab = color.rgb2lab(image)
                image_lab = (color.rgb2lab(image) + [0,128,128]) #// [1,1,1]
                #print("===== IMAGE LAB =====\n {}".format(image_lab))

                # loop over the number of segments
                # apply SLIC and extract (approximately) the supplied number of segments
                gt_boundaries, gt_segmentation = helper._get_groundtruth(path_groundtruths+filename[:-4]+".mat")
                # labels with 0 are ignored, seems legit? --- ENFORCE STARTING AT 1, START_LABEL IS FUCKING DUMP
                if method == "SLIC":
                    if(argsy['read'] == "True"):
                        Gr = nx.read_gpickle(path_pickles+filename[:-4]+".pkl")
                        labels = helper._loadlabels(path_labels+filename[:-4]+".preseg")
                    else:
                        labels = 1+slic(image, n_segments = _num_segments, compactness=_compactness, convert2lab=True, start_label=0)
                        # WARNING: NEVER USE WITH DISTANCE 0
                        G = graph.RAG(labels,connectivity=2)
                        distance=4
                        Gr = helper._distance_r_graph(G,distance,image_lab,labels,threshold=0.,sigma=_sigma)
                    _dt=2.9
                # FIXME: LOAD GRAPH WITH NETWORKX FOR BETTER COMPUTING TIME
                elif method == "MSP":
                    if(argsy['read'] == "True"):
                        Gr = nx.read_gpickle(path_pickles+filename[:-4]+".pkl")
                        labels = pickle.load(open(path_labels+filename[:-4]+".preseg","rb"))
                    else:
                        labels = helper._meanshift_py(dirpath+filename,_spatial_radius,_range_radius,_min_density)
                        Gr = graph.rag_mean_color(image_lab,labels,connectivity=2,mode='similarity',sigma=_sigma)
                        #adjacency, _= _get_Lab_adjacency(labels,image_lab,_sigma)
                        #G = graph.RAG(labels,connectivity=2)
                        #Gr = _distance_zero_graph(G,image_lab,adjacency,threshold=0.)
                    _dt=1.2
                else:
                    ms_image = cv2.imread(dirpath+filename)
                    labels = _meanshift_opencv(image_lab)
                    #connectivity 2 means 8 neighbors
                    Gr = graph.rag_mean_color(image_lab,labels,connectivity=2,mode='similarity',sigma=_sigma)
                    _dt=1.2
                        
                number_regions = numpy.amax(labels)
            
                primsp = helper._probabilistic_rand_index(gt_segmentation,labels)
                
                '''n2v = Node2Vec.from_nx_graph(Gr)
                n2v.simulate_walks(
                walk_length=80,
                n_walks=10,
                p=2.5,
                q=0.2,
                workers=4,
                verbose=False,
                #rand_seed=SEED
                )

                n2v.learn_embeddings(
                    dimensions=8,
                    context_size=5,
                    epochs=2,
                    workers=4,
                    verbose=False,
                    #rand_seed=SEED
                )
                
                embn2v = n2v.embeddings
                bothemb = n2v.embeddings.tolist()'''
                
                #feature_vector = _color_features(labels,image_lab)
                #for l,node in enumerate(Gr.nodes()):
                    #embn2v.append(model.wv.get_vector(str(node)).tolist())
                    #bothemb.append(model.wv.get_vector(str(node)).tolist())
                
                labels_merged,has_merged=merge_pixels(labels,image_lab,thr_pixels=300,sigma=_sigma)
                segmentation,has_merged=merge_cosine(labels_merged,image_lab,thr=0.997,sigma=_sigma)
                number_regions_merged = numpy.amax(segmentation)
            
                pri = helper._probabilistic_rand_index(gt_segmentation,segmentation)
                
                AVG_GEST_PRI.append(pri)
                MSP_PRI.append(primsp)
                
                tmpmis, tmpvoi, tmpari =[], [], []
                mspmis, mspvoi, mspari =[], [], []
                for l in range(len(gt_segmentation)):
                    tmpvoi.append(sum(variation_of_information(gt_segmentation[l].flatten(),segmentation.flatten())))
                    tmpmis.append(normalized_mutual_info_score(gt_segmentation[l].flatten(),segmentation.flatten()))
                    tmpari.append(helper._rand_index_score(gt_segmentation[l].flatten(), segmentation.flatten()))
                    mspvoi.append(sum(variation_of_information(gt_segmentation[l].flatten(),labels.flatten())))
                    mspmis.append(normalized_mutual_info_score(gt_segmentation[l].flatten(),labels.flatten()))
                    mspari.append(helper._rand_index_score(gt_segmentation[l].flatten(), labels.flatten()))
                VOI.append(mean(tmpvoi))
                MIS.append(mean(tmpmis))
                VOIMSP.append(mean(mspvoi))
                MISMSP.append(mean(mspmis))
                        
                print("AVG GeST PRI:",number_regions,number_regions_merged,primsp,pri,mean(MSP_PRI), mean(AVG_GEST_PRI))
                print("MEAN GeST VOI:",mean(VOIMSP),mean(VOI))
                print("MEAN GeST MIS:",mean(MISMSP),mean(MIS))
                    
            if(write): 
                pickle.dump(labels,open(path_labels+str(i+1)+"_"+filename[:-4]+".preseg","wb"))
                pickle.dump(best_segmentation,open(path_labels+str(i+1)+"_"+filename[:-4]+".seg","wb"))
                pickle.dump(clusterings, open(path_clusterings+str(i+1)+"_"+filename[:-4]+".clt","wb"))
                numpy.save(path_embeddings+filename[:-4]+".emb",bothemb)
                nx.write_gpickle(Gr, path_pickles+str(i+1)+"_"+filename[:-4]+".pkl")
                nx.write_weighted_edgelist(Gr, path_graphs+filename[:-4]+".wgt", delimiter='\t')
                helper._savepreseg(labels, image, path_presegs+filename[:-4]+".png")
                helper._savefig(best_segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+".png")
                
                
                    
