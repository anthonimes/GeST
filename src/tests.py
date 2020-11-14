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
    dirpath,_,images = list(walk(path_images))[0]

    for _thr in range(10):
        PRI, VOI = [], []
        for i,filename in enumerate(sorted(images)):
            has_merged=False
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
            
            # pretty good with .5 2 80 10
            # computing embeddings with Node2vec framework
            Gn2v = node2vec.Graph(Gr, False, 8, .1)
            Gn2v.preprocess_transition_probs()
            walks = Gn2v.simulate_walks(80, 10)
            model=learn_embeddings(walks,dimensions=16)
            
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
                
            segmentation = numpy.zeros(labels.shape,dtype=int)
        
            # silhouette method to detect best number of clusters
            if(silh):
                selected_k = min(silhouette(datagest,25),number_regions)
            else:
                selected_k = min(n_cluster,number_regions)
                
            # using agglomerative clustering to obtain segmentation 
            clustering = cl.AgglomerativeClustering(n_clusters=selected_k,affinity='cosine',linkage='average',distance_threshold=None).fit(datagest)
            labels_clustering = clustering.labels_
            # building flat segmentation and then reshaping
            segmentation=numpy.asarray([labels_clustering[value-1]+1 for line in labels for value in line]).reshape(labels.shape)
            #for l,line in enumerate(labels):
            #    for j,value in enumerate(line):
            #        segmentation[l][j] = labels_clustering[value-1]+1
                
            pri = helper._probabilistic_rand_index(gt_segmentation,segmentation)
            tmpvoi = [sum(variation_of_information(gt_segmentation[l].flatten(),segmentation.flatten())) for l in range(len(gt_segmentation))]
            print(filename,selected_k,number_regions,pri,mean(tmpvoi),end=' ')
            
            # merging hard images with empirical thresholds
            #if(filename in hardimages):
            if(True):
                #helper._savefig(segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+"_"+str(selected_k)+"_NOT_MERGED.png")
                segmentation,has_merged=helper._merge(segmentation,image_lab,thr_pixels=250,thr=0.998,sigma=_sigma)
                
                #segmentation,has_merged=helper._merge_cosine(segmentation,image_lab,thr=0.998,sigma=_sigma)
                #segmentation,has_merged=helper._merge_pixels(segmentation,image_lab,thr_pixels=300,sigma=_sigma)
                        
                pri = helper._probabilistic_rand_index(gt_segmentation,segmentation)
                #helper._savefig(segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+"_"+str(selected_k)+"_"+str(numpy.amax(segmentation))+".png")
                tmpvoi = [sum(variation_of_information(gt_segmentation[l].flatten(),segmentation.flatten())) for l in range(len(gt_segmentation))]
                
            print(has_merged,pri,mean(tmpvoi))
            PRI.append(pri)
            VOI.append(mean(tmpvoi))
                    
            if(write): 
                pickle.dump(labels,open(path_labels+str(i+1)+"_"+filename[:-4]+".preseg","wb"))
                pickle.dump(segmentation,open(path_labels+str(i+1)+"_"+filename[:-4]+".seg","wb"))
                #pickle.dump(clusterings, open(path_clusterings+str(i+1)+"_"+filename[:-4]+".clt","wb"))
                numpy.save(path_embeddings+filename[:-4]+".emb",gestemb)
                nx.write_gpickle(Gr, path_pickles+str(i+1)+"_"+filename[:-4]+".pkl")
                nx.write_weighted_edgelist(Gr, path_graphs+filename[:-4]+".wgt", delimiter='\t')
                #helper._savepreseg(labels, image, path_presegs+filename[:-4]+".png")
                helper._savefig(segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+"_"+str(selected_k)+".png")
                    
        GEST_VOI.append(mean(VOI))
        GEST_PRI.append(mean(PRI))
        print(GEST_PRI, GEST_VOI, max(GEST_PRI),mean(GEST_PRI),mean(GEST_VOI))
