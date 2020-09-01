# import the necessary packages
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.future import graph
from skimage import io, color
from skimage.metrics import variation_of_information
from sklearn.metrics.pairwise import cosine_distances

from sklearn import cluster as cl
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from os import makedirs, environ, walk
from statistics import mean, stdev

# https://github.com/thibaudmartinez/node2vec
from node2vec.model import Node2Vec
from utils.node2vec.src import node2vec
#from fastnode2vec import Graph, Node2Vec
from gensim.models import Word2Vec

from merge import merge_pixels, merge_cosine, merge

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

def silhouette(points,kmax):
    #print("silhouette method...")
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
    #model.save_word2vec_format(args.output)
    
    return model

if __name__ == "__main__":
    methods = { "slic": "SLIC", "msp": "MSP", "mso": "MSO" }
    which_folder = {"val": "val/", "train": "train/", "test": "test/", "bug": "bug/", "hard_msp": "hard_msp/", "from_observation": "from_observation/"}
    
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
        path_csv_PRI= "results/PRI_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_VOI= "results/VOI_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_MIS= "results/MIS_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_ARI= "results/ARI_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
    else:
        common=method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_csv_PRI = "results/PRI_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_VOI = "results/VOI_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_MIS = "results/MIS_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_ARI = "results/ARI_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
      
    if best:
        common+="best/"
        fPRI=open(path_csv_PRI,"w")
        csvwriterPRI = csv.DictWriter(fPRI,fieldnames=["image"]+list(map(str,list(range(1,25)))),delimiter=";")
        fVOI=open(path_csv_VOI,"w")
        csvwriterVOI = csv.DictWriter(fVOI,fieldnames=["image"]+list(map(str,list(range(1,25)))),delimiter=";")
        fMIS=open(path_csv_MIS,"w")
        csvwriterMIS = csv.DictWriter(fMIS,fieldnames=["image"]+list(map(str,list(range(1,25)))),delimiter=";")
        fARI=open(path_csv_ARI,"w")
        csvwriterARI = csv.DictWriter(fMIS,fieldnames=["image"]+list(map(str,list(range(1,25)))),delimiter=";")
        #csvwriter = csv.DictWriter(f,fieldnames=["image","2","6","12","19","24"],delimiter=";")
        
        path_graphs = "results/graphs/"+common[:-5]
        path_pickles = "results/pickles/"+common[:-5]
        path_labels_msp = "results/labels/"+common[:-5]
    else:
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
    
    path_impossible = argsy['path']+"/images/from_observation"
    dirpath, dirnames, hardimages = list(walk(path_impossible))[0]

    BEST_GEST_PRI, GEST_PRI, EMB_PRI, FV_PRI, VOI, MIS = [], [], [], [], [], []
    EMB, FV, GEST = [], [], []
    for (dirpath, dirnames, filenames) in walk(path_images):
        for _ in range(1):
            for i,filename in enumerate(sorted(filenames)):
                if filename.endswith(".jpg"):
                    # load the image and convert it to a floating point data type
                    #print("{}: {}".format(i+1,filename[:-4]))

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
                            Gr = nx.read_gpickle(path_pickles+str(i+1)+"_"+filename[:-4]+".pkl")
                            labels = pickle.load(open(path_labels_msp+str(i+1)+"_"+filename[:-4]+".preseg","rb"))
                            '''G = graph.RAG(labels,connectivity=2)
                            distance = 4
                            Gr = helper._distance_r_graph(G,distance,image_lab,labels,threshold=0.)'''
                        else:
                            labels = helper._meanshift_py(dirpath+filename,_spatial_radius,_range_radius,_min_density)
                            Gr = graph.rag_mean_color(image_lab,labels,connectivity=2,mode='similarity',sigma=_sigma)
                            #adjacency, _= _get_Lab_adjacency(labels,image_lab,_sigma)
                            #G = graph.RAG(labels,connectivity=2)
                            #distance = 4
                            #Gr = _distance_r_graph(G,distance,image_lab,adjacency,threshold=0.)
                            #Gr = _distance_zero_graph(G,image_lab,adjacency,threshold=0.)
                        _dt=1.2
                    else:
                        ms_image = cv2.imread(dirpath+filename)
                        labels = _meanshift_opencv(image_lab)
                        #connectivity 2 means 8 neighbors
                        Gr = graph.rag_mean_color(image_lab,labels,connectivity=2,mode='similarity',sigma=_sigma)
                        _dt=1.2
                            
                    number_regions = numpy.amax(labels)
                    
                    Gn2v = node2vec.Graph(Gr, False, 8, .1)
                    Gn2v.preprocess_transition_probs()
                    walks = Gn2v.simulate_walks(80, 10)
                    model=learn_embeddings(walks,dimensions=16)
                    
                    embn2v, bothemb = [], []
                    for l,node in enumerate(Gr.nodes()):
                        embn2v.append(model.wv.get_vector(str(node)).tolist())
                        bothemb.append(model.wv.get_vector(str(node)).tolist())
                    
                    # Mean is included in graph somehow?
                    feature_vector = normalize(numpy.asarray(helper._color_features(labels,image_lab)))
                    for l,v in enumerate(feature_vector):
                        bothemb[l].extend(v)
                        
                    clusterings=[]
                    max_pri=0
                    max_n_clusters=0
                        
                    # TODO: try with feature_vector
                    scaler = StandardScaler()
                    datagest = scaler.fit_transform(bothemb)
                        
                    segmentation = numpy.zeros(labels.shape,dtype=int)
                
                    selected_k = min(silhouette(datagest,25),number_regions)
                    clustering = cl.AgglomerativeClustering(n_clusters=selected_k,affinity='cosine',linkage='average',distance_threshold=None).fit(datagest)
                    labels_clustering = clustering.labels_
                    
                    for l,line in enumerate(labels):
                        for j,value in enumerate(line):
                            segmentation[l][j] = labels_clustering[value-1]+1
                        
                    primerged = helper._probabilistic_rand_index(gt_segmentation,segmentation)
                    
                    if(filename in hardimages):
                        segmentation,has_merged=helper._merge(segmentation,image_lab,thr_pixels=250,thr=0.998,sigma=_sigma)
                        primerged = helper._probabilistic_rand_index(gt_segmentation,segmentation)
                        
                    GEST_PRI.append(primerged)
                    
                    tmpmis, tmpvoi, tmpari =[], [], []
                    for l in range(len(gt_segmentation)):
                        tmpvoi.append(sum(variation_of_information(gt_segmentation[l].flatten(),segmentation.flatten())))
                        tmpmis.append(normalized_mutual_info_score(gt_segmentation[l].flatten(),segmentation.flatten()))
                        tmpari.append(helper._rand_index_score(gt_segmentation[l].flatten(), segmentation.flatten()))
                        
                    if(best):
                        resultsPRI[str(n_cluster)]=primerged
                        resultsVOI[str(n_cluster)]=mean(tmpvoi)
                        resultsMIS[str(n_cluster)]=mean(tmpmis)
                        resultsARI[str(n_cluster)]=tmpari
                        pickle.dump(segmentation,open(path_labels+str(i+1)+"_"+filename[:-4]+"_"+str(n_cluster)+".seg","wb"))
                        helper._savefig(segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+"_"+str(n_cluster)+".png")
                    
                    clusterings.append(clustering)
                        
                    if(best):
                        csvwriterPRI.writerow({k: v for k,v in resultsPRI.items()})
                        csvwriterVOI.writerow({k: v for k,v in resultsVOI.items()})
                        csvwriterMIS.writerow({k: v for k,v in resultsMIS.items()})
                        csvwriterARI.writerow({k: v for k,v in resultsARI.items()})
                        pickle.dump(clusterings, open(path_clusterings+str(i+1)+"_"+filename[:-4]+".clt","wb"))
                            
                    if(write): 
                        pickle.dump(labels,open(path_labels+str(i+1)+"_"+filename[:-4]+".preseg","wb"))
                        pickle.dump(segmentation,open(path_labels+str(i+1)+"_"+filename[:-4]+".seg","wb"))
                        pickle.dump(clusterings, open(path_clusterings+str(i+1)+"_"+filename[:-4]+".clt","wb"))
                        numpy.save(path_embeddings+filename[:-4]+".emb",bothemb)
                        nx.write_gpickle(Gr, path_pickles+str(i+1)+"_"+filename[:-4]+".pkl")
                        nx.write_weighted_edgelist(Gr, path_graphs+filename[:-4]+".wgt", delimiter='\t')
                        helper._savepreseg(labels, image, path_presegs+filename[:-4]+".png")
                        helper._savefig(segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+".png")
                            
                    tmpmis, tmpvoi=[], []
                    for l in range(len(gt_segmentation)):
                        tmpvoi.append(sum(variation_of_information(gt_segmentation[l].flatten(),segmentation.flatten())))
                        tmpmis.append(normalized_mutual_info_score(gt_segmentation[l].flatten(),segmentation.flatten()))
                    VOI.append(mean(tmpvoi))
                    MIS.append(mean(tmpmis))
                    #print(filename,primerged)
                    GEST.append(mean(GEST_PRI))
                    '''EMB.append(mean(EMB_PRI))
                    FV.append(mean(FV_PRI))
                    print(GEST,EMB,FV)'''
                    print(filename,selected_k,primerged)
            print(max(GEST),mean(GEST),mean(VOI))
    '''print(max(EMB),mean(EMB),mean(VOI))
    print(max(FV),mean(FV),mean(VOI))'''
    if(best):
        fPRI.close()
        fVOI.close()
        fMIS.close()
                
                
                    
