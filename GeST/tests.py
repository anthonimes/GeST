# import the necessary packages
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.future import graph
from skimage import io, color
from skimage.metrics import variation_of_information

from sklearn import cluster as cl
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score

from os import makedirs, environ, walk
from statistics import mean, stdev

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
        path_csv_GESTPRI= "results/GEST_PRI_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_GESTVOI= "results/GEST_VOI_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_GESTMIS= "results/GEST_MIS_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_EMBPRI= "results/EMB_PRI_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_EMBVOI= "results/EMB_VOI_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_EMBMIS= "results/EMB_MIS_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_BOTHPRI= "results/BOTH_PRI_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_BOTHVOI= "results/BOTH_VOI_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_BOTHMIS= "results/BOTH_MIS_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_MSPPRI= "results/MSP_PRI_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_MSPVOI= "results/MSP_VOI_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_MSPMIS= "results/MSP_MIS_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
        #path_csv_ARI= "results/ARI_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+".csv"
    else:
        common=method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_csv_GESTPRI = "results/GEST_PRI_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_GESTVOI = "results/GEST_VOI_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_GESTMIS = "results/GEST_MIS_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_EMBPRI = "results/EMB_PRI_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_EMBVOI = "results/EMB_VOI_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_EMBMIS = "results/EMB_MIS_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_BOTHPRI = "results/BOTH_PRI_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_BOTHVOI = "results/BOTH_VOI_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_BOTHMIS = "results/BOTH_MIS_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_MSPPRI = "results/MSP_PRI_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_MSPVOI = "results/MSP_VOI_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
        path_csv_MSPMIS = "results/MSP_MIS_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
        #path_csv_ARI = "results/ARI_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"
      
    if best:
        common+="best/"
        fGESTPRI=open(path_csv_GESTPRI,"w")
        csvwriterGESTPRI = csv.DictWriter(fGESTPRI,fieldnames=["image"]+list(map(str,list(range(1,25)))),delimiter=";")
        fGESTVOI=open(path_csv_GESTVOI,"w")
        csvwriterGESTVOI = csv.DictWriter(fGESTVOI,fieldnames=["image"]+list(map(str,list(range(1,25)))),delimiter=";")
        fGESTMIS=open(path_csv_GESTMIS,"w")
        csvwriterGESTMIS = csv.DictWriter(fGESTMIS,fieldnames=["image"]+list(map(str,list(range(1,25)))),delimiter=";")
        fEMBPRI=open(path_csv_EMBPRI,"w")
        csvwriterEMBPRI = csv.DictWriter(fEMBPRI,fieldnames=["image"]+list(map(str,list(range(1,25)))),delimiter=";")
        fEMBVOI=open(path_csv_EMBVOI,"w")
        csvwriterEMBVOI = csv.DictWriter(fEMBVOI,fieldnames=["image"]+list(map(str,list(range(1,25)))),delimiter=";")
        fEMBMIS=open(path_csv_EMBMIS,"w")
        csvwriterEMBMIS = csv.DictWriter(fEMBMIS,fieldnames=["image"]+list(map(str,list(range(1,25)))),delimiter=";")
        fBOTHPRI=open(path_csv_BOTHPRI,"w")
        csvwriterBOTHPRI = csv.DictWriter(fBOTHPRI,fieldnames=["image"]+list(map(str,list(range(1,25)))),delimiter=";")
        fBOTHVOI=open(path_csv_BOTHVOI,"w")
        csvwriterBOTHVOI = csv.DictWriter(fBOTHVOI,fieldnames=["image"]+list(map(str,list(range(1,25)))),delimiter=";")
        fBOTHMIS=open(path_csv_BOTHMIS,"w")
        csvwriterBOTHMIS = csv.DictWriter(fBOTHMIS,fieldnames=["image"]+list(map(str,list(range(1,25)))),delimiter=";")
        fMSPPRI=open(path_csv_MSPPRI,"w")
        csvwriterMSPPRI = csv.DictWriter(fMSPPRI,fieldnames=["image","MSP"],delimiter=";")
        fMSPVOI=open(path_csv_MSPVOI,"w")
        csvwriterMSPVOI = csv.DictWriter(fMSPVOI,fieldnames=["image", "MSP"],delimiter=";")
        fMSPMIS=open(path_csv_MSPMIS,"w")
        csvwriterMSPMIS = csv.DictWriter(fMSPMIS,fieldnames=["image", "MSP"],delimiter=";")
        #fARI=open(path_csv_ARI,"w")
        #csvwriterARI = csv.DictWriter(fMIS,fieldnames=["image"]+list(map(str,list(range(1,25)))),delimiter=";")
        #csvwriter = csv.DictWriter(f,fieldnames=["image","2","6","12","19","24"],delimiter=";")
        
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
    
    path_impossible = argsy['path']+"/images/from_observation"
    dirpath, dirnames, hardimages = list(walk(path_impossible))[0]

    GESTPRI, EMBPRI, BOTHPRI, MSPPRI, VOI, MIS, MSPVOI, MSPMIS, EMBVOI, EMBMIS, BOTHVOI, BOTHMIS = [], [], [], [], [], [], [], [], [], [], [], []
    for (dirpath, dirnames, filenames) in walk(path_images):
        for i,filename in enumerate(sorted(filenames)):
            if filename.endswith(".jpg"):
                print("{}: {}".format(i+1,filename), end=' ')
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
                
                Gn2v = node2vec.Graph(Gr, False, 4, .2)
                Gn2v.preprocess_transition_probs()
                walks = Gn2v.simulate_walks(80, 10)
                model=learn_embeddings(walks,dimensions=16)
                
                embn2v, bothemb = [], []
                for l,node in enumerate(Gr.nodes()):
                    embn2v.append(model.wv.get_vector(str(node)).tolist())
                    bothemb.append(model.wv.get_vector(str(node)).tolist())
                
                '''n2v = Node2Vec.from_nx_graph(Gr)
                n2v.simulate_walks(
                walk_length=80,
                n_walks=10,
                p=4,
                q=0.2,
                workers=4,
                verbose=False,
                rand_seed=SEED
                )

                n2v.learn_embeddings(
                    dimensions=8,
                    context_size=5,
                    epochs=2,
                    workers=4,
                    verbose=False,
                    rand_seed=SEED
                )
                
                embn2v = n2v.embeddings
                bothemb = n2v.embeddings.tolist()'''
                
                #feature_vector = _color_features(labels,image_lab)
                #for l,node in enumerate(Gr.nodes()):
                    #embn2v.append(model.wv.get_vector(str(node)).tolist())
                    #bothemb.append(model.wv.get_vector(str(node)).tolist())
                
                feature_vector = normalize(numpy.asarray(helper._color_features(labels,image_lab)))
                
                for l,v in enumerate(feature_vector):
                    bothemb[l].extend(v)
                    
                scaler = StandardScaler()
                datagest = scaler.fit_transform(embn2v)
                datafv = scaler.fit_transform(feature_vector)
                databoth = scaler.fit_transform(bothemb)
                    
                clusterings=[]
                if best:
                    #NUM_CLUSTERS = [2,6,12,19,24]
                    NUM_CLUSTERS = list(range(2,25))
                    resultsGESTPRI = {"image": filename[:-4]}
                    resultsGESTVOI = {"image": filename[:-4]}
                    resultsGESTMIS = {"image": filename[:-4]}
                    resultsEMBPRI = {"image": filename[:-4]}
                    resultsEMBMIS = {"image": filename[:-4]}
                    resultsEMBVOI = {"image": filename[:-4]}
                    resultsBOTHMIS = {"image": filename[:-4]}
                    resultsBOTHVOI = {"image": filename[:-4]}
                    resultsBOTHPRI = {"image": filename[:-4]}
                    resultsMSPPRI = {"image": filename[:-4]}
                    resultsMSPVOI = {"image": filename[:-4]}
                    resultsMSPMIS = {"image": filename[:-4]}
                else:
                    # NOTE: TRY WITH A HIGH NUMBER OF CLUSTERS+MERGE
                    NUM_CLUSTERS = [min(24,number_regions)]
                
                max_pri=0
                max_n_clusters=0
                
                primsp = helper._probabilistic_rand_index(gt_segmentation, labels)
                mspmis, mspvoi = [], []
                for l in range(len(gt_segmentation)):
                    mspvoi.append(sum(variation_of_information(gt_segmentation[l].flatten(),labels.flatten())))
                    mspmis.append(normalized_mutual_info_score(gt_segmentation[l].flatten(),labels.flatten()))
                MSPVOI.append(mean(mspvoi))
                MSPMIS.append(mean(mspmis))
                if(best):
                    resultsMSPPRI["MSP"]=primsp
                    resultsMSPVOI["MSP"]=mean(mspvoi)
                    resultsMSPMIS["MSP"]=mean(mspmis)
                    csvwriterMSPPRI.writerow({k: v for k,v in resultsMSPPRI.items()})
                    csvwriterMSPVOI.writerow({k: v for k,v in resultsMSPVOI.items()})
                    csvwriterMSPMIS.writerow({k: v for k,v in resultsMSPMIS.items()})
                
                for n_cluster in NUM_CLUSTERS:
                    gestmis, gestvoi, embmis, embvoi, bothmis, bothvoi = [], [], [], [], [], []
                    
                    # NODE2VEC EMBEDDINGS
                    segmentation_gest = numpy.zeros(labels.shape,dtype=int)
                
                    selected_k = min(n_cluster,number_regions)
                    clustering_gest = cl.AgglomerativeClustering(n_clusters=selected_k,affinity='cosine',linkage='average',distance_threshold=None).fit(datagest)
                    labels_clustering_gest = clustering_gest.labels_
                    
                    for l,line in enumerate(labels):
                        for j,value in enumerate(line):
                            segmentation_gest[l][j] = labels_clustering_gest[value-1]+1
                    
                    # FEATURE VECTOR EMBEDDINGS
                    segmentation_fv = numpy.zeros(labels.shape,dtype=int)
                
                    selected_k = min(n_cluster,number_regions)
                    clustering_fv = cl.AgglomerativeClustering(n_clusters=selected_k,affinity='cosine',linkage='average',distance_threshold=None).fit(datafv)
                    labels_clustering_fv = clustering_fv.labels_
                    
                    for l,line in enumerate(labels):
                        for j,value in enumerate(line):
                            segmentation_fv[l][j] = labels_clustering_fv[value-1]+1
                    
                    # BOTH EMBEDDINGS
                    segmentation_both = numpy.zeros(labels.shape,dtype=int)
                
                    selected_k = min(n_cluster,number_regions)
                    clustering_both = cl.AgglomerativeClustering(n_clusters=selected_k,affinity='cosine',linkage='average',distance_threshold=None).fit(databoth)
                    labels_clustering_both = clustering_both.labels_
                    
                    for l,line in enumerate(labels):
                        for j,value in enumerate(line):
                            segmentation_both[l][j] = labels_clustering_both[value-1]+1
                            
                    # trying to merge hard images
                    '''if(filename in hardimages):
                        
                        labels_merged_gest,has_merged=merge_pixels(segmentation_gest,image_lab,thr_pixels=200,sigma=_sigma)
                        segmentation_gest,has_merged=merge_cosine(labels_merged_gest,image_lab,thr=0.997,sigma=_sigma)
                        labels_merged_fv,has_merged=merge_pixels(segmentation_fv,image_lab,thr_pixels=200,sigma=_sigma)
                        segmentation_fv,has_merged=merge_cosine(labels_merged_fv,image_lab,thr=0.997,sigma=_sigma)
                        labels_merged_both,has_merged=merge_pixels(segmentation_both,image_lab,thr_pixels=200,sigma=_sigma)
                        segmentation_both,has_merged=merge_cosine(labels_merged_both,image_lab,thr=0.997,sigma=_sigma)'''
                    
                    prigest = helper._probabilistic_rand_index(gt_segmentation,segmentation_gest)
                    priemb = helper._probabilistic_rand_index(gt_segmentation,segmentation_fv)
                    priboth = helper._probabilistic_rand_index(gt_segmentation,segmentation_both)
                    
                    if(prigest > max_pri):
                        max_pri=prigest
                        best_segmentation=segmentation_gest
                        best_clustering=labels_clustering_gest
                    
                    #tmpmis, tmpvoi, tmpari =[], [], []
                    for l in range(len(gt_segmentation)):
                        gestvoi.append(sum(variation_of_information(gt_segmentation[l].flatten(),segmentation_gest.flatten())))
                        gestmis.append(normalized_mutual_info_score(gt_segmentation[l].flatten(),segmentation_gest.flatten()))
                        embvoi.append(sum(variation_of_information(gt_segmentation[l].flatten(),segmentation_fv.flatten())))
                        embmis.append(normalized_mutual_info_score(gt_segmentation[l].flatten(),segmentation_fv.flatten()))
                        bothvoi.append(sum(variation_of_information(gt_segmentation[l].flatten(),segmentation_both.flatten())))
                        bothmis.append(normalized_mutual_info_score(gt_segmentation[l].flatten(),segmentation_both.flatten()))
                        #tmpari.append(helper._rand_index_score(gt_segmentation[l].flatten(), segmentation.flatten()))
                        
                    if(best):
                        resultsGESTPRI[str(n_cluster)]=prigest
                        resultsGESTVOI[str(n_cluster)]=mean(gestvoi)
                        resultsGESTMIS[str(n_cluster)]=mean(gestmis)
                        #resultsEMBARI[str(n_cluster)]=tmpari
                        resultsEMBPRI[str(n_cluster)]=priemb
                        resultsEMBVOI[str(n_cluster)]=mean(embvoi)
                        resultsEMBMIS[str(n_cluster)]=mean(embmis)
                        resultsBOTHPRI[str(n_cluster)]=priboth
                        resultsBOTHVOI[str(n_cluster)]=mean(bothvoi)
                        resultsBOTHMIS[str(n_cluster)]=mean(bothmis)
                        
                    '''if(write):
                        pickle.dump(segmentation_gest,open(path_labels+"GeST_"+str(i+1)+"_"+filename[:-4]+"_"+str(selected_k)+".seg","wb"))
                        pickle.dump(segmentation_fv,open(path_labels+"FV_"+str(i+1)+"_"+filename[:-4]+"_"+str(selected_k)+".seg","wb"))
                        pickle.dump(segmentation_both,open(path_labels+"BOTH_"+str(i+1)+"_"+filename[:-4]+"_"+str(selected_k)+".seg","wb"))
                        helper._savefig(segmentation_gest, image, path_figs+"GeST_"+str(i+1)+"_"+filename[:-4]+"_"+str(selected_k)+".png")
                        helper._savefig(segmentation_fv, image, path_figs+"FV_"+str(i+1)+"_"+filename[:-4]+"_"+str(selected_k)+".png")
                        helper._savefig(segmentation_both, image, path_figs+"BOTH_"+str(i+1)+"_"+filename[:-4]+"_"+str(selected_k)+".png")
                        helper._savefig(best_segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+".png")'''
                        
                    clusterings.append(labels_clustering_gest)
                    
                print(max_pri)
                        
                if(best):
                    csvwriterGESTPRI.writerow({k: v for k,v in resultsGESTPRI.items()})
                    csvwriterGESTVOI.writerow({k: v for k,v in resultsGESTVOI.items()})
                    csvwriterGESTMIS.writerow({k: v for k,v in resultsGESTMIS.items()})
                    #csvwriterARI.writerow({k: v for k,v in resultsARI.items()})
                    csvwriterEMBPRI.writerow({k: v for k,v in resultsEMBPRI.items()})
                    csvwriterEMBVOI.writerow({k: v for k,v in resultsEMBVOI.items()})
                    csvwriterEMBMIS.writerow({k: v for k,v in resultsEMBMIS.items()})
                    csvwriterBOTHPRI.writerow({k: v for k,v in resultsBOTHPRI.items()})
                    csvwriterBOTHVOI.writerow({k: v for k,v in resultsBOTHVOI.items()})
                    csvwriterBOTHMIS.writerow({k: v for k,v in resultsBOTHMIS.items()})
                        
                if(write): 
                    pickle.dump(best_segmentation,open(path_labels+"GeST_"+str(i+1)+"_"+filename[:-4]+"_"+str(selected_k)+".seg","wb"))
                    helper._savefig(best_segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+".png")
                    pickle.dump(labels,open(path_labels+str(i+1)+"_"+filename[:-4]+".preseg","wb"))
                    pickle.dump(clusterings, open(path_clusterings+str(i+1)+"_"+filename[:-4]+".clt","wb"))
                    numpy.save(path_embeddings+filename[:-4]+"_BOTH.emb",bothemb)
                    numpy.save(path_embeddings+filename[:-4]+".emb",embn2v)
                    nx.write_gpickle(Gr, path_pickles+str(i+1)+"_"+filename[:-4]+".pkl")
                    nx.write_weighted_edgelist(Gr, path_graphs+filename[:-4]+".wgt", delimiter='\t')
                    helper._savepreseg(labels, image, path_presegs+filename[:-4]+".png")
                
    if(best):
        fGESTPRI.close()
        fGESTVOI.close()
        fGESTMIS.close()
        fEMBPRI.close()
        fEMBVOI.close()
        fEMBMIS.close()
        fBOTHPRI.close()
        fBOTHVOI.close()
        fBOTHMIS.close()
        fMSPPRI.close()
        fMSPVOI.close()
        fMSPMIS.close()
                
                
                    
