# import the necessary packages
from skimage.segmentation import slic, quickshift
from skimage.util import img_as_float
from skimage.future import graph
from skimage import io,color,measure
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn import cluster as cl
from sklearn.metrics import silhouette_score
from skimage.segmentation import felzenszwalb
from skimage.metrics import (adapted_rand_error,
                              variation_of_information,mean_squared_error)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from utils.parse_matlab import get_groundtruth, _savemat
from utils.graph import _distance_r_graph, _get_Lab_adjacency, _color_features, _distance_zero_graph
from utils import save
from utils.metrics.pri import probabilistic_rand_index

# https://github.com/thibaudmartinez/node2vec
from node2vec.model import Node2Vec
from utils.node2vec.src import node2vec
#from fastnode2vec import Graph, Node2Vec
from gensim.models import Word2Vec

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

from os import walk, environ, makedirs
from statistics import mean, stdev
from math import sqrt
import numpy, csv
import networkx as nx
import warnings, sys, argparse
warnings.filterwarnings("ignore")

# https://github.com/fjean/pymeanshift
import pymeanshift as pms
# used by pymeanshift
import cv2, pickle

# for reproducibility
SEED = 42
environ["PYTHONHASHSEED"] = str(SEED)

def learn_embeddings(walks,dimensions=32,window_size=10,min_count=0,workers=4,iter=1):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''

    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=min_count, sg=1, workers=workers, iter=iter)
    #model.save_word2vec_format(args.output)
    
    return model

def _meanshift_py(path,_sr,_rr,_mind):
    ms_image = cv2.imread(path)
    (segmented_image, labels, number_regions) = pms.segment(ms_image, spatial_radius=_sr, range_radius=_rr, min_density=_mind)
    return 1+labels

# https://stackoverflow.com/questions/46392904/scikit-mean-shift-algorithm-returns-black-picture
def _meanshift_opencv(image_lab):
    #Loading original image
    #originImg = cv2.imread(path)
    # Shape of original image
    originShape = image_lab.shape

    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities
    flatImg=numpy.reshape(image_lab, [-1, 3])

    # Estimate bandwidth for meanshift algorithm
    bandwidth = cl.estimate_bandwidth(flatImg, quantile=0.04, n_samples=100)
    ms = cl.MeanShift(bandwidth = bandwidth, bin_seeding=True)

    # Performing meanshift on flatImg
    ms.fit(flatImg)

    # (r,g,b) vectors corresponding to the different clusters after meanshift
    labels=ms.labels_
    return (labels+1).reshape(originShape[:-1])

def elbow(points,kmax):
    #print("elbow method...")

    # function returns WSS score for k values from 1 to kmax
    def WSS():
        ssd = []
        for k in range(1, kmax+1):
            #scaler = StandardScaler()
            #data = scaler.fit_transform(points)
            km = cl.KMeans(n_clusters=k,random_state=10)
            km = km.fit(points)
            ssd.append(km.inertia_)
        return ssd
    
    from kneed import KneeLocator
    
    scores = WSS()
    x = range(1,len(scores)+1)
    kn = KneeLocator(x, scores, curve='convex', direction='decreasing')
    #print("done. \nbest number of clusters: {}".format(kn.knee))

    return kn.knee

def silhouette(points,kmax):
    #print("silhouette method...")
    def SSE():
        sse=[]
        for k in range(2, kmax):
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(points) + (k + 1) * 10])

            #km = cl.KMeans(init = "k-means++", n_clusters = k, n_init = 35, random_state=10)
            #km = km.fit(points) 
            km = cl.AgglomerativeClustering(n_clusters=k,affinity='cosine',linkage='average',distance_threshold=None).fit(points)
            labels_clustering = km.labels_
            silhouette_avg=silhouette_score(points, labels_clustering, metric = 'cosine')
            sse.append(silhouette_avg)
            
            '''labels_from_clustering = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
            for i,line in enumerate(labels):
                for j,value in enumerate(line):
                    # +1 needed since 0=background
                    labels_from_clustering[i][j] = labels_clustering[value-1]+1
            print(probabilistic_rand_index(GT,labels_from_clustering),silhouette_avg)

            y_lower = 10
            sample_silhouette_values = silhouette_samples(points, labels_clustering)
            for i in range(k):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[labels_clustering == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / k)
                ax1.fill_betweenx(numpy.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(labels_clustering.astype(float) / k)
            ax2.scatter(points[:, 0], points[:, 1], marker='.', s=50, lw=0, alpha=0.7, c=colors, edgecolor='k')

            # Labeling the clusters
            centers = km.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % k),
                         fontsize=14, fontweight='bold')
            plt.show()'''
        return sse
        
    scores = SSE()
    best = scores.index(max(scores))+2

    return best

def modularity(G,points,kmax):
    #print("silhouette method...")
    def MOD():
        mod=[]
        for k in range(2, kmax):
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(points) + (k + 1) * 10])

            km = cl.KMeans(init = "k-means++", n_clusters = k, n_init = 35, random_state=10)
            km = km.fit(points)
            labels = km.labels_
            partition = dict()
            for i,node in enumerate(G.nodes()):
                # assigning cluster as community for every node
                partition[node] = labels[node-1]
            mod.append(community_louvain.modularity(partition,G))
        return mod
        
    scores = MOD()
    x = range(len(scores))
    best = scores.index(max(scores))+2

    return best

def _apply_clustering(NUM_CLUSTER, embeddings,image,labels,method="",groundtruth=None,GT=None,verbose=False,_dt=2,best=False,_k=0,path=None,filename=None):
    #for NUM_CLUSTER in NUM_CLUSTERS:
    # IMPACT ON AGGLOMERATIVE?
    data=embeddings
    '''scaler = StandardScaler()
    data = scaler.fit_transform(embeddings)
    variance = 0.98 #The higher the explained variance the more accurate the model will remain, but more dimensions will be present
    pca = PCA(variance)

    pca.fit(data) #fit the data according to our PCA instance
    #print("Number of components before PCA  = " + str(data.shape[1]))
    #print("Number of components after PCA 0.98 = " + str(pca.n_components_))
    data = pca.transform(data)'''
    #print("Dimension of our data after PCA = " + str(data.shape))
    #clustering = cl.KMeans(init = "k-means++", n_clusters = NUM_CLUSTER, n_init = 35, random_state=10)
    #clustering.fit(data)
    #clustering = cl.MeanShift().fit(embeddings)
    #clustering = cl.KMeans(n_clusters=NUM_CLUSTER,random_state=10).fit(data)
    #clustering.fit(embeddings)
    #clustering = cl.DBSCAN(eps=0.5,metric='euclidean',n_jobs=4).fit(data)
    if best is False:
        clustering = cl.AgglomerativeClustering(n_clusters=_k,affinity='cosine',linkage='average',distance_threshold=None).fit(embeddings)
        labels_clustering = clustering.labels_
        labels_from_clustering = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
        for i,line in enumerate(labels):
            for j,value in enumerate(line):
                # +1 needed since 0=background
                labels_from_clustering[i][j] = labels_clustering[value-1]+1
        pri = probabilistic_rand_index(gt_segmentation,labels_from_clustering)
        
        clustering = cl.AgglomerativeClustering(n_clusters=None,affinity='euclidean',linkage='average',distance_threshold=_dt).fit(embeddings)
        labels_clustering = clustering.labels_
        labels_from_clustering = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
        for i,line in enumerate(labels):
            for j,value in enumerate(line):
                # +1 needed since 0=background
                labels_from_clustering[i][j] = labels_clustering[value-1]+1
        thre_pri = probabilistic_rand_index(gt_segmentation,labels_from_clustering)

        '''clustering = cl.KMeans(init = "k-means++", n_clusters = _k, n_init = 35, random_state=10)
        clustering.fit(embeddings)
        labels_clustering = clustering.labels_
        labels_from_clustering = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
        for i,line in enumerate(labels):
            for j,value in enumerate(line):
                # +1 needed since 0=background
                labels_from_clustering[i][j] = labels_clustering[value-1]+1
        pri = probabilistic_rand_index(gt_segmentation,labels_from_clustering)'''
        return pri,thre_pri, labels_from_clustering, numpy.amax(labels_from_clustering)
    else:
        sse = []
        mean_pri = []
        max_pri,max_pri_dt=0,0
        best_segmentation = None
        clusterings = []
        for k in NUM_CLUSTERS:
            clustering = cl.AgglomerativeClustering(n_clusters=k,affinity='cosine',linkage='average',distance_threshold=None).fit(embeddings)
            #clustering = cl.KMeans(init = "k-means++", n_clusters = k, n_init = 35, random_state=10)
            #clustering.fit(data)
            labels_clustering = clustering.labels_
            clusterings.append(labels_clustering)
            labels_from_clustering = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
            for i,line in enumerate(labels):
                for j,value in enumerate(line):
                    # +1 needed since 0=background
                    labels_from_clustering[i][j] = labels_clustering[value-1]+1

            pri = probabilistic_rand_index(gt_segmentation,labels_from_clustering)
            mean_pri.append(pri)
            if(pri > max_pri):
                max_pri = pri
                best_segmentation=labels_from_clustering
        #print("mean {} \t stdev {} \t max {} {}\t min {}".format(mean(mean_pri),stdev(mean_pri),max_pri,max(mean_pri),min(mean_pri)))
        pickle.dump(clusterings, open(path+filename+".clt","wb"))
        return max_pri,max_pri,best_segmentation, numpy.amax(best_segmentation)

if __name__ == "__main__":

    methods = { "slic": "SLIC", "msp": "MSP", "mso": "MSO" }
    which_folder = {"val": "val/", "train": "train/", "test": "test/", "hard_msp": "hard_msp/", "impossible": "impossible/"}
    
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = False, help = "Path to the image")
    ap.add_argument("-p", "--path", required = False, help = "Path to folder")
    ap.add_argument("-m", "--method", required = True, help="pre-segmentation method")
    ap.add_argument("-b", "--best", required = False, help="compute best clustering?", default=False)
    ap.add_argument("-w", "--write", required = False, help="write all files to hard drive?", default=False)
    ap.add_argument("-d", "--dataset", required = False, help="which of {train,val,test} to evaluate?", default="val")
    ap.add_argument("-r", "--read", required = False, help="which of {train,val,test} to evaluate?", default=False)
    ap.add_argument("--hs", required = False, help="spatial radius?", default=15)
    ap.add_argument("--hr", required = False, help="range radius?", default=4.5)
    ap.add_argument( "--mind", required = False, help="min density", default=300)
    ap.add_argument( "--sigma", required = True, help="kernel parameter", default=50)
    ap.add_argument( "--segments", required = False, help="number of segments (SLIC)", default=50)
    ap.add_argument( "--compactness", required = False, help="compactness (SLIC)", default=50)

    argsy = vars(ap.parse_args())
    method = methods[argsy['method']]
    # computing best clustering ?
    argsy['best'] = True if argsy['best'] == "True" else False
    argsy['write'] = True if argsy['write'] == "True" else False
    folder = which_folder[argsy['dataset']]
    
    path_images = argsy['path']+"/images/"+folder
    path_groundtruths = argsy['path']+"/groundTruth/"+folder

    _spatial_radius=int(argsy['hs']) #hs
    _range_radius=float(argsy['hr']) #hr
    _min_density=int(argsy['mind'])
    _sigma=float(argsy['sigma'])
    _num_segments = float(argsy['segments'])
    _compactness = float(argsy['compactness'])
    
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
        path_csv= "results/best_scores_clusterings_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+".csv"

    if argsy['best'] is True:
        f=open(path_csv,"w")
        # TODO: FOR SLIC, SAME CREATION BUT IF NUMBER_REGIONS < 25 THEN APPEND ZEROS
        #csvwriter = csv.DictWriter(f,fieldnames=["image"]+list(map(str,list(range(1,25)))),delimiter=";")
        csvwriter = csv.DictWriter(f,fieldnames=["image","2","6","12","19","24"],delimiter=";")
        path_graphs+="best/"
        path_pickles+="best/"
        path_labels+="best/"
        path_figs+="best/"
        path_presegs+="best/"
        path_embeddings+="best/"
        path_clusterings+="best/"
        path_matlab+="best/"

    makedirs(path_graphs,exist_ok=True)
    makedirs(path_pickles,exist_ok=True)
    makedirs(path_labels,exist_ok=True)
    makedirs(path_scores,exist_ok=True)
    makedirs(path_figs,exist_ok=True)
    makedirs(path_presegs,exist_ok=True)
    makedirs(path_embeddings,exist_ok=True)
    makedirs(path_clusterings,exist_ok=True)
    makedirs(path_matlab,exist_ok=True)

    BEST_GEST_PRI, AVG_GEST_PRI, EMB_PRI, BOTH_PRI, BOTH_SCALED, VOI = [], [], [], [], [], []
    # load the image and convert it to a floating point data type
    for (dirpath, dirnames, filenames) in walk(path_images):
        for i,filename in enumerate(sorted(filenames)):
            if filename.endswith(".jpg"):
                print("{}: {}".format(i+1,filename),end=' ')
                image = io.imread(dirpath+filename)
                image = img_as_float(image)

                #image_lab = image
                image_lab = color.rgb2lab(image)
                image_lab = (color.rgb2lab(image) + [0,128,128]) #// [1,1,1]
                #print("===== IMAGE LAB =====\n {}".format(image_lab))

                # loop over the number of segments
                # apply SLIC and extract (approximately) the supplied number of segments
                gt_boundaries, gt_segmentation = get_groundtruth(path_groundtruths+filename[:-4]+".mat")
                # labels with 0 are ignored, seems legit? --- ENFORCE STARTING AT 1, START_LABEL IS FUCKING DUMP
                if method == "SLIC":
                    if(argsy['read'] == "True"):
                        Gr = nx.read_gpickle(path_pickles+filename[:-4]+".pkl")
                        labels = save._loadlabels(path_labels+filename[:-4]+".preseg")
                    else:
                        labels = 1+slic(image, n_segments = _num_segments, compactness=_compactness, convert2lab=True, start_label=0)
                        adjacency,_ = _get_Lab_adjacency(labels,image_lab,_sigma)
                        # WARNING: NEVER USE WITH DISTANCE 0
                        G = graph.RAG(labels,connectivity=2)
                        distance=4
                        Gr = _distance_r_graph(G,distance,image_lab,adjacency,threshold=0.)
                    _dt=2.9
                    if argsy['best'] is True:
                        NUM_CLUSTERS = [2,6,12,19,24]
                        #NUM_CLUSTERS = list(range(1,25))
                        results = {"image": filename[:-4]}
                    else:
                        NUM_CLUSTERS = [19]
                # FIXME: LOAD GRAPH WITH NETWORKX FOR BETTER COMPUTING TIME
                elif method == "MSP":
                    if(argsy['read'] == "True"):
                        Gr = nx.read_gpickle(path_pickles+filename[:-4]+".pkl")
                        labels = pickle.load(open(path_labels+filename[:-4]+".preseg","rb"))
                    else:
                        labels = _meanshift_py(dirpath+filename,_spatial_radius,_range_radius,_min_density)
                        Gr = graph.rag_mean_color(image_lab,labels,connectivity=2,mode='similarity',sigma=_sigma)
                        #adjacency, _= _get_Lab_adjacency(labels,image_lab,_sigma)
                        #G = graph.RAG(labels,connectivity=2)
                        #Gr = _distance_zero_graph(G,image_lab,adjacency,threshold=0.)
                    _dt=1.2
                    if argsy['best'] is True:
                        #NUM_CLUSTERS = [2,6,12,19,24,30]
                        NUM_CLUSTERS = list(range(1,25))
                        results = {"image": filename[:-4]}
                    else:
                        NUM_CLUSTERS = [19]
                else:
                    ms_image = cv2.imread(dirpath+filename)
                    labels = _meanshift_opencv(image_lab)
                    #connectivity 2 means 8 neighbors
                    Gr = graph.rag_mean_color(image_lab,labels,connectivity=2,mode='similarity',sigma=_sigma)
                    _dt=1.2
                        
                number_regions = numpy.amax(labels)
                segmentation = numpy.zeros(labels.shape,dtype=int)
                    
                '''Gc=Gr.copy()
                to_remove=[e for e in Gr.edges(data='weight') if e[2]<=0.9]
                Gc.remove_edges_from(to_remove)
                connected_component_Gc = sorted(nx.connected_components(Gc), key=len, reverse=True)[:1]
                sizes = [len(cc) for cc in connected_component_Gc]
                print("graph with {} connected components of size {}".format(len(connected_component_Gc), sizes), end=' ')
                
                labels_cc = numpy.zeros(numpy.amax(labels),dtype=int)
                label=1
                print(connected_component_Gc)
                for cc in connected_component_Gc:
                    for vertex in cc:
                        labels_cc[vertex-1]=label
                    label+=1
                    
                for l,line in enumerate(labels):
                    for j,value in enumerate(line):
                        # +1 needed since 0=background
                        if (value-1) in connected_component_Gc[0]:
                            segmentation[l][j] = labels_cc[value-1]
                    
                save._savefig(segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+"_CC.png")
                
                Gr = Gr.subgraph(list(Gr.nodes()-(1+numpy.asarray(list(connected_component_Gc[0])))))
                #Gcc=Gr'''
                            
                max_pri=0
                max_n_clusters=0
                maxseg=0
                
                '''Gn2v = node2vec.Graph(Gr, False, 4, .5)
                Gn2v.preprocess_transition_probs()
                walks = Gn2v.simulate_walks(80, 10)
                model=learn_embeddings(walks,dimensions=32)'''
                
                n2v = Node2Vec.from_nx_graph(Gr)
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
                bothemb = n2v.embeddings.tolist()
                
                dictemb = dict()
                #feature_vector = _color_features(labels,image_lab)
                #for l,node in enumerate(Gr.nodes()):
                    #embn2v.append(model.wv.get_vector(str(node)).tolist())
                    #bothemb.append(model.wv.get_vector(str(node)).tolist())
                    #dictemb[node]=l
                #segmentation=numpy.zeros(labels.shape,dtype=int)
                
                feature_vector = normalize(numpy.asarray(_color_features(labels,image_lab)))
                
                for l,v in enumerate(feature_vector):
                    bothemb[l].extend(v)
                    
                scaler = StandardScaler()
                data = scaler.fit_transform(bothemb)
                #data = embn2v
                    
                clusterings=[]
                
                for n_cluster in NUM_CLUSTERS:
                    
                    if(n_cluster<number_regions):
                
                        selected_k = min(n_cluster,number_regions)
                        clustering = cl.AgglomerativeClustering(n_clusters=selected_k,affinity='cosine',linkage='average',distance_threshold=None).fit(data)
                        #clustering = cl.KMeans(init = "k-means++", n_clusters = selected_k, n_init = 35, random_state=10).fit(data)
                        labels_clustering = clustering.labels_
                        
                        for l,line in enumerate(labels):
                            for j,value in enumerate(line):
                                #if (value) in Gr.nodes():
                                    # +1 needed since 0=background
                                    #segmentation[l][j] = labels_clustering[dictemb[value]]+1
                                segmentation[l][j] = labels_clustering[value-1]+1
                                
                        pri = probabilistic_rand_index(gt_segmentation,segmentation)
                        if(argsy['best'] is True):
                            results[str(n_cluster)]=pri
                        
                        AVG_GEST_PRI.append(pri)
                        if(pri > max_pri):
                            max_pri=pri
                            best_segmentation=segmentation
                            best_clustering=labels_clustering
                        
                        clusterings.append(clustering)
                        
                    else:
                        if(argsy['best'] is True):
                            results[str(n_cluster)]=0
                        
                    
                    if(argsy['best']):
                        pickle.dump(segmentation,open(path_labels+str(i+1)+"_"+filename[:-4]+"_"+str(selected_k)+".seg","wb"))
                        #save._savelabels(labels,path_labels+filename[:-4]+".preseg")
                        #save._savelabels(best_segmentation,path_labels+filename[:-4]+".seg")
                        #save._savefig(segmentation, image, path_figs+filename[:-4]+"_"+str(i+1)+".png",colored=True)
                
                if(argsy['best']):
                    csvwriter.writerow({k: v for k,v in results.items()})
                        
                if(argsy['write']): 
                    pickle.dump(labels,open(path_labels+str(i+1)+"_"+filename[:-4]+".preseg","wb"))
                    pickle.dump(best_segmentation,open(path_labels+str(i+1)+"_"+filename[:-4]+".seg","wb"))
                    pickle.dump(clusterings, open(path_clusterings+str(i+1)+"_"+filename[:-4]+".clt","wb"))
                    nx.write_gpickle(Gr, path_pickles+str(i+1)+"_"+filename[:-4]+".pkl")
                    nx.write_weighted_edgelist(Gr, path_graphs+filename[:-4]+".wgt", delimiter='\t')
                    save._savepreseg(labels, image, path_presegs+filename[:-4]+".png")
                    save._saveembeddings(embn2v,path_embeddings+filename[:-4]+".emb")
                    save._savefig(best_segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+".png")
                        
                    # FIXME: they use five segmentations for their tests (according to thresholds) --> USE 5 FUCKING CLUSTERINGS 
                #save._savefig(best_segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+"_"+str(n_cluster)+".png")  
                #_savemat(path_matlab+filename[:-4]+".mat",best_segmentation)
                BEST_GEST_PRI.append(max_pri)
                #tmpmse, tmpvoi=[], []
                #for l in range(len(gt_segmentation)):
                #    tmpvoi.append(sum(variation_of_information(gt_segmentation[l],segmentation)))
                #VOI.append(mean(tmpvoi))
                print("BEST GeST PRI:",number_regions,max_pri,mean(BEST_GEST_PRI))
                print("AVG GeST PRI:",number_regions,mean(AVG_GEST_PRI))
                
    if(argsy['best']):
        f.close()
                
                
                    
