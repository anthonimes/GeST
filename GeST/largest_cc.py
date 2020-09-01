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

from utils.parse_matlab import get_groundtruth, get_BSR
from utils.graph import _distance_r_graph, _get_Lab_adjacency, _color_features
from utils import save
from utils.metrics.pri import probabilistic_rand_index

# https://github.com/thibaudmartinez/node2vec
from node2vec.model import Node2Vec
from utils.node2vec.src import node2vec
#from fastnode2vec import Graph, Node2Vec
from gensim.models import Word2Vec

from os import walk, environ, makedirs
from statistics import mean, stdev
from math import sqrt
import numpy
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

def merge(labels,image_lab,similarity,thr_pixels=200,thr_similarity=0.58):
    def _mean_lab():
        mean_lab = []
        for index,region in enumerate(regions):
            # getting coordinates of region
            coords = region.coords
            #cy, cx = region.centroid
            #plt.plot(cx, cy, 'ro')
            #plt.show()
            L_value, a_value, b_value=[],[],[]
            for (x,y) in coords:
                L,a,b=image_lab[(x,y)]
                L_value.append(L)
                a_value.append(a)
                b_value.append(b)
            mean_lab.append((mean(L_value),mean(a_value),mean(b_value)))
        return mean_lab
    # NOTE; labels must be a matrix-like imaeg
    labels_merge = numpy.copy(labels)
    merged=True
    has_merged=False
    while(merged):
        regions = measure.regionprops(labels_merge)
        print(len(regions))
        G = graph.RAG(labels_merge,connectivity=2)
        merged=False
        mean_lab = _mean_lab()
        print(len(mean_lab))
        for i in range(len(regions)):
            for j in range(i+1,len(regions)):
                Ri = regions[i]
                Rj = regions[j]
                lenRi = len(Ri.coords)
                lenRj = len(Rj.coords)
                if(G.has_edge(Ri.label,Rj.label)):
                    if(lenRi < thr_pixels) or (lenRj < thr_pixels):
                        max_label = Ri if Ri.label > Rj.label else Rj
                        min_label = Ri if Ri.label < Rj.label else Rj
                        for (x,y) in max_label.coords:
                            labels_merge[(x,y)] = min_label.label
                            merged=True
                            has_merged=True
                            # updating remaining labels
                        for i,line in enumerate(labels):
                            for j,value in enumerate(line):
                                if value > max_label.label:
                                    labels_merge[i][j] -= 1
                    else:
                        print(len(mean_lab),Ri.label-1,Rj.label-1)
                        dL=(mean_lab[Ri.label-1][0]-mean_lab[Rj.label-1][0])**2
                        da=(mean_lab[Ri.label-1][1]-mean_lab[Rj.label-1][1])**2
                        db=(mean_lab[Ri.label-1][2]-mean_lab[Rj.label-1][2])**2
                        sim = sqrt(dL+da+db)
                        if sim <= thr_similarity:
                            max_label = Ri if Ri.label > Rj.label else Rj
                            min_label = Ri if Ri.label < Rj.label else Rj
                            for (x,y) in max_label.coords:
                                labels_merge[(x,y)] = min_label.label
                                merged=True
                                has_merged=True
                            # updating remaining labels
                            for i,line in enumerate(labels_merge):
                                for j,value in enumerate(line):
                                    if value > max_label.label:
                                        labels_merge[i][j] -= 1
                if(merged):
                    break
            if(merged):
                break
    return labels_merge,has_merged


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

    if argsy['best'] is True:
        path_graphs+="best/"
        path_pickles+="best/"
        path_labels+="best/"
        path_figs+="best/"
        path_presegs+="best/"
        path_embeddings+="best/"
        path_clusterings+="best/"

    makedirs(path_graphs,exist_ok=True)
    makedirs(path_pickles,exist_ok=True)
    makedirs(path_labels,exist_ok=True)
    makedirs(path_scores,exist_ok=True)
    makedirs(path_figs,exist_ok=True)
    makedirs(path_presegs,exist_ok=True)
    makedirs(path_embeddings,exist_ok=True)
    makedirs(path_clusterings,exist_ok=True)

    GEST_PRI, EMB_PRI, BOTH_PRI, BOTH_SCALED = [], [], [], []
    # load the image and convert it to a floating point data type
    for (dirpath, dirnames, filenames) in walk(path_images):
        for i,filename in enumerate(sorted(filenames)):
            if filename.endswith(".jpg"):
                print("{}: {}".format(i+1,filename),end=' ')
                image = io.imread(dirpath+filename)
                image = img_as_float(image)
                ms_image = cv2.imread(dirpath+filename)

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
                        adjacency = _get_Lab_adjacency(labels,image_lab,_sigma)
                        G = graph.RAG(labels,connectivity=2)
                        distance=2
                        Gr = _distance_r_graph(G,distance,image_lab,adjacency,threshold=0.)
                    _dt=2.9
                # FIXME: LOAD GRAPH WITH NETWORKX FOR BETTER COMPUTING TIME
                elif method == "MSP":
                    if(argsy['read'] == "True"):
                        Gr = nx.read_gpickle(path_pickles+filename[:-4]+".pkl")
                        labels = save._loadlabels(path_labels+filename[:-4]+".preseg")
                    else:
                        labels = _meanshift_py(dirpath+filename,_spatial_radius,_range_radius,_min_density)
                        regions=measure.regionprops(labels)
                        #Gr = graph.rag_mean_color(image_lab,labels,connectivity=2,mode='similarity',sigma=_sigma)
                        adjacency, feature_vector = _get_Lab_adjacency(labels,image_lab,_sigma)
                        G = graph.RAG(labels,connectivity=2)
                        distance=0
                        Gr = _distance_r_graph(G,distance,image_lab,adjacency,threshold=0.)
                    _dt=1.2
                else:
                    labels = _meanshift_opencv(image_lab)
                    #connectivity 2 means 8 neighbors
                    Gr = graph.rag_mean_color(image_lab,labels,connectivity=2,mode='similarity',sigma=_sigma)
                    _dt=1.2
                        
                number_regions = numpy.amax(labels)
                    
                # We enforce the largest isolated component calculated by MSP
                Gc=Gr.copy()
                to_remove=[e for e in Gr.edges(data='weight') if e[2]<=0.58]
                Gc.remove_edges_from(to_remove)
                connected_component_Gc = sorted(nx.connected_components(Gc), key=len, reverse=False)
                filter(lambda x: len(x)<2,connected_component_Gc)
                connected_component_Gc=[list(e)[0] for e in connected_component_Gc]
                sizes = [(regions[cc-1].label,len(regions[cc-1].coords)) for cc in connected_component_Gc]
                #print("graph with {} connected components of size {}".format(len(connected_component_Gc)), end=' ')
                connected_component_Gc=max(sizes,key=lambda x: x[1])[0]
                    
                segmentation = numpy.zeros(labels.shape,dtype=int)
                for l,line in enumerate(labels):
                    for j,value in enumerate(line):
                        # +1 needed since 0=background
                        if value==connected_component_Gc:
                            segmentation[l][j] = 1
                    
                save._savefig(segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+"_CC.png")
                
                Gr = Gr.subgraph(list(Gr.nodes()-({connected_component_Gc})))
                #Gcc=Gr
                
                max_pri=0
                max_n_clusters=0
                maxseg=0
                
                Gn2v = node2vec.Graph(Gr, False, 1, 0.2)
                Gn2v.preprocess_transition_probs()
                walks = Gn2v.simulate_walks(40, 20)
                model=learn_embeddings(walks,dimensions=32)
                embn2v,bothemb=[],[]
                dictemb = dict()
                #feature_vector = _color_features(labels,image_lab)
                for l,node in enumerate(Gr.nodes()):
                    embn2v.append(model.wv.get_vector(str(node)).tolist())
                    bothemb.append(model.wv.get_vector(str(node)).tolist())
                    dictemb[node]=l
                
                clustering = cl.AgglomerativeClustering(n_clusters=min(24,number_regions),affinity='cosine',linkage='average',distance_threshold=None).fit(embn2v)
                labels_clustering = clustering.labels_
                
                maxseg=numpy.amax(segmentation)
                for l,line in enumerate(labels):
                    for j,value in enumerate(line):
                        if (value) in Gr.nodes():
                            # +1 needed since 0=background
                            segmentation[l][j] = labels_clustering[dictemb[value]]+maxseg+1
                        #segmentation[l][j] = labels_clustering[value-1]+maxseg+1
                            
                max_pri = probabilistic_rand_index(gt_segmentation,segmentation)
                GEST_PRI.append(max_pri)
                print("GeST PRI:",number_regions,max_pri,mean(GEST_PRI))
                save._savefig(segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+"_GEST.png")  
                
                '''feature_vector = _color_features(labels,image_lab)
                
                clustering = cl.AgglomerativeClustering(n_clusters=min(24,number_regions),affinity='cosine',linkage='average',distance_threshold=None).fit(feature_vector)
                labels_clustering = clustering.labels_
                
                for l,line in enumerate(labels):
                    for j,value in enumerate(line):
                        #if (value) in Gcc.nodes():
                            # +1 needed since 0=background
                            #segmentation[l][j] = labels_clustering[dictemb[value]]+maxseg+1
                        segmentation[l][j] = labels_clustering[value-1]+maxseg+1
                            
                max_pri = probabilistic_rand_index(gt_segmentation,segmentation)
                EMB_PRI.append(max_pri)
                print("EMB PRI:",number_regions,max_pri,mean(EMB_PRI))
                save._savefig(segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+"_EMB.png") 
                
                clustering = cl.AgglomerativeClustering(n_clusters=min(24,number_regions),affinity='cosine',linkage='average',distance_threshold=None).fit(bothemb)
                labels_clustering = clustering.labels_
                
                for l,line in enumerate(labels):
                    for j,value in enumerate(line):
                        #if (value) in Gcc.nodes():
                            # +1 needed since 0=background
                            #segmentation[l][j] = labels_clustering[dictemb[value]]+maxseg+1
                        segmentation[l][j] = labels_clustering[value-1]+maxseg+1
                            
                max_pri = probabilistic_rand_index(gt_segmentation,segmentation)
                BOTH_PRI.append(max_pri)
                print("BOTH PRI:",number_regions,max_pri,mean(BOTH_PRI))
                save._savefig(segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+"_BOTH.png")''' 
                
                #feature_vector = numpy.asarray(_color_features(labels,image_lab))
                
                '''for l,v in enumerate(feature_vector):
                    bothemb[l].extend(v)
                
                scaler = StandardScaler()
                data = scaler.fit_transform(bothemb)
                
                clustering = cl.AgglomerativeClustering(n_clusters=min(24,number_regions),affinity='cosine',linkage='average',distance_threshold=None).fit(data)
                labels_clustering = clustering.labels_
                
                for l,line in enumerate(labels):
                    for j,value in enumerate(line):
                        #if (value) in Gcc.nodes():
                            # +1 needed since 0=background
                            #segmentation[l][j] = labels_clustering[dictemb[value]]+maxseg+1
                        segmentation[l][j] = labels_clustering[value-1]+maxseg+1
                            
                max_pri = probabilistic_rand_index(gt_segmentation,segmentation)
                BOTH_SCALED.append(max_pri)
                print("BOTH SCALED:",number_regions,max_pri,mean(BOTH_SCALED))
                save._savefig(segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+"_SCALED.png") 
                
                #maxseg=numpy.amax(segmentation)
                SCORES=[]
                for k in range(1,min(25,len(Gcc))):
                    segmentation=numpy.zeros(labels.shape,dtype=int)
                    clustering = cl.AgglomerativeClustering(n_clusters=k,affinity='cosine',linkage='average',distance_threshold=None).fit(embn2v)
                    labels_clustering = clustering.labels_
                    
                    for l,line in enumerate(labels):
                        for j,value in enumerate(line):
                            if (value) in Gcc.nodes():
                                # +1 needed since 0=background
                                #segmentation[l][j] = labels_clustering[dictemb[value]]+maxseg+1
                                segmentation[l][j] = labels_clustering[value-1]+maxseg+1
                       
                       
                    pri = probabilistic_rand_index(gt_segmentation,segmentation)
                    pri1 = probabilistic_rand_index(gt_segmentation,segmentation)
                    pri2 = probabilistic_rand_index(gt_segmentation,segmentation)
                    print(pri,pri1,pri2)
                    SCORES.append(pri)
                    if(pri>max_pri):
                        max_pri=pri
                        max_n_clusters=k
                        best_segmentation=segmentation
                            
                    save._savefig(segmentation, image, path_figs+str(i+1)+"_"+filename[:-4]+"_"+str(k)+".png")                
                    labels_emb = numpy.zeros(labels_cc.shape,dtype=int)
                    for l,node in enumerate(Gr.nodes()):
                        labels_emb[node-1]=labels_clustering[dictemb[node]]+maxseg+1
                    all_labels=labels_cc+labels_emb'''
                #pricc=probabilistic_rand_index(gt_segmentation,best_segmentation)
                #print("PRI CC",number_regions,pricc,mean(CC))#,community_louvain.modularity(partition_GCC,Gmc))
                #BSL_PRI.append(probabilistic_rand_index(gt_segmentation,labels))

                #labels_felzenszwalb = felzenszwalb(image,scale=300)
                #FELZ_PRI.append(probabilistic_rand_index(gt_segmentation,labels_felzenszwalb))
                #labels = 1+quickshift(image, kernel_size=2, max_dist=6, ratio=0.5)
                #print("number of regions from MeanShift: {}".format(number_regions))

                '''save._savefig(labels,image,path_results+"SLIC/"+filename[:-4]+".png")            

                for i in range(len(gt_segmentation)):
                    save._savefig(gt_segmentation[i], image, path_results+"/groundtruth/images/"+filename[:-4]+"_"+str(i+1)+".png",colored=True)
                    save._savelabels(gt_segmentation[i],path_results+"/groundtruth/labels/"+filename[:-4]+"_"+str(i+1)+".lbl")
                    save._savelabels_seg(gt_segmentation[i],path_results+"/groundtruth/segmentation/"+filename[:-4]+"_"+str(i+1)+".seg",filename)'''

                #Gr = nx.read_weighted_edgelist(path_graphs+filename[:-4]+".wgt", nodetype=int)
                #labels = save._loadlabels(path_labels+filename[:-4]+".lbl")
                #print("number of regions {}:".format(number_regions))

                #adjacency=_get_histo_adjacency(labels,image_lab,dirpath,filename)

                # FIXME: clean the graph using a threshold to gain time!
                '''lc = graph.show_rag(labels,Gr,image)      
                lcc = graph.show_rag(labels,Gc,image)

                fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

                ax[0].set_title('Gr RAG')

                lc = graph.show_rag(labels, Gr, image,
                                    img_cmap='gray', edge_cmap='viridis', ax=ax[1])

                ax[1].set_title('Gc RAG')
                lc = graph.show_rag(labels, Gc, image,
                                    img_cmap='gray', edge_cmap='viridis', ax=ax[1])
                for a in ax:
                    a.axis('off')

                plt.tight_layout()
                plt.show()

                # computing distance-R graph 
                #connected = "connected" if nx.is_connected(Gr) else "not connected"
                #pairs=(len(Gr)*(len(Gr)-1))/2
                #print("generated {} vertices and {} edges {} graph ({} pairs)".format(len(Gr),len(Gr.edges()),connected,pairs))

                # FIXME: augment features with image features
                # Node2Vec
                # FIXME: QUEL EST LE PUTAIN DE FUCK SANS DECONNER ???
                # FIXED: IT SEEMS I FOUND A STABLE IMPLEMENTATION, AT LAST!!!!!
                # https://github.com/thibaudmartinez/node2vec

                # NOTE: try InfoMap + nfm (also, a graph going further could be useful)
                # FUCK ME I'M FAMOUS: does not work if the graph is not connected ??!
                # FIXME: embeddings should be different depending on graph/method used!
                edges=[(str(i),str(j)) for (i,j) in Gr.edges()]
                edges=[(x[0][0],x[0][1],x[1]) for x in list(zip(edges,[e[2] for e in Gr.edges(data='weight')]))]
                Gn2v= Graph(edges,directed=False, weighted=True)
                n2v = Node2Vec(Gn2v, dim=16, walk_length=40, context=5, p=1.0, q=0.2, workers=4)
                n2v.train(epochs=100)
                embn2v=[]
                for node in Gr.nodes():
                    embn2v.append(n2v.wv.get_vector(str(node)).tolist())
                # Simulate biased random walks on the graph
                
                node2vec = Node2Vec(Gr, dimensions=32, walk_length=20, num_walks=10, workers=1)  # Use temp_folder for big graphs
                model = node2vec.fit(window=5, min_count=1)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
                embn2v = []
                for node in Gr.nodes():
                    embn2v.append(model.wv.get_vector(str(node)))
                feature_vector = _color_features(labels,image_lab)
                embeddings = [0]*len(embn2v)
                for i in range(len(embn2v)):
                    embeddings[i]=embn2v[i].tolist()+feature_vector[i]'''
                '''max_pri = 0
                max_nb_clusters=0
                best_k=0
                #best_k = elbow(embn2v,min(25,number_regions))
                if(number_regions)>1:
                    if(argsy['best'] is False):
                        k = silhouette(numpy.asarray(embn2v),min(25,number_regions),gt_segmentation,labels)
                        #k = modularity(Gr,numpy.asarray(embn2v),min(25,number_regions))
                    else:
                        k=0
                    k_pri,thre_pri,segmentation,max_nb_clusters=_apply_clustering(NUM_CLUSTERS,numpy.asarray(embn2v),image,numpy.asarray(labels),GT=gt_segmentation,_dt=_dt,best=argsy['best'],_k=k,path=path_clusterings,filename=filename[:-4])
                else:
                    k_pri,max_nb_clusters=probabilistic_rand_index(gt_segmentation,labels),1
                tmpmse, tmpvoi=[], []
                for i in range(len(gt_segmentation)):
                    tmpvoi.append(sum(variation_of_information(gt_segmentation[i],segmentation)))
                    tmpmse.append(mean_squared_error(gt_segmentation[i],segmentation))
                VOI.append(mean(tmpvoi))
                MSE.append(mean(tmpmse))
                GEST_PRI.append(1-k_pri)
                EMB_PRI.append(thre_pri)
                print("GeST:",max_nb_clusters,k_pri,thre_pri,1-k_pri,mean(tmpmse),mean(tmpvoi),mean(GEST_PRI),mean(VOI),mean(MSE),mean(EMB_PRI))
                #print("GeST:",nb_clusters,pri,mean(GEST_PRI))'''
                if(argsy['write'] == "True"):
                    nx.write_gpickle(Gr, path_pickles+filename[:-4]+".pkl")
                    nx.write_weighted_edgelist(Gr, path_graphs+filename[:-4]+".wgt", delimiter='\t')
                    save._savelabels(labels,path_labels+filename[:-4]+".preseg")
                    save._savepreseg(labels, image, path_presegs+filename[:-4]+".png")
                    save._saveembeddings(embn2v,path_embeddings+filename[:-4]+".emb")
                    #save._savefig(segmentation, image, path_figs+filename[:-4]+"_"+str(i+1)+".png",colored=True)
                    save._savelabels(segmentation,path_labels+filename[:-4]+".seg")
    #print("CC PRI: {} {}".format(mean(GEST_PRI),stdev(GEST_PRI))) 
    #print("SLIC PRI: {} {}".format(mean(BSL_PRI),stdev(BSL_PRI))) 
    #print("GeST PRI: {} {}".format(mean(GEST_PRI),stdev(GEST_PRI)))
    #print("Felz PRI: {} {}".format(mean(FELZ_PRI),stdev(FELZ_PRI)))
    
    
    
    
    
                        
    # We remove every vertex from the largest component(S ?)
    
    #Gcc = Gmc.subgraph(list(Gmc.nodes()-(1+numpy.asarray(list(connected_component_Gr[0])))))
    #distance=4
    #adjacency=_get_Lab_adjacency(labels,image_lab,sigma=_sigma)
    #Gcc = _distance_r_graph(Gcc,distance,adjacency=adjacency,threshold=0.58)
    
    '''connected_component_Gcc = sorted(nx.connected_components(Gcc), key=len, reverse=True)
    sizes = [len(cc) for cc in connected_component_Gcc]
    print("graph with {} connected components of size {}".format(len(connected_component_Gcc), sizes))'''
    
    '''
    segmentation_merged=None
    merged=True
    has_merged=False
    print("starting from... {}".format(numpy.amax(segmentation.flatten())))
    while(merged):
        regions = measure.regionprops(segmentation_merged)
        Gmc = graph.rag_mean_color(image_lab,segmentation_merged,connectivity=2,mode='similarity',sigma=_sigma)
    
        adjacency = nx.to_numpy_matrix(Gmc)
        adjacency[adjacency<=0.99005]=0
        Gmc = nx.from_numpy_matrix(adjacency)
        merged=False
        
        for u,v,sim in Gmc.edges(data='weight'):
            Ri = regions[u]
            Rj = regions[v]
            lenRi = len(Ri.coords)
            lenRj = len(Rj.coords)
            if(lenRi < 200) or (lenRj < 200):
                max_label = Ri if Ri.label > Rj.label else Rj
                min_label = Ri if Ri.label < Rj.label else Rj
                for (x,y) in max_label.coords:
                    segmentation_merged[(x,y)] = min_label.label
                    merged=True
                    has_merged=True
                    # updating remaining labels
                    for l,line in enumerate(labels):
                        for j,value in enumerate(line):
                            if value > max_label.label:
                                segmentation_merged[l][j] -= 1
            else:
                max_label = Ri if Ri.label > Rj.label else Rj
                min_label = Ri if Ri.label < Rj.label else Rj
                for (x,y) in max_label.coords:
                    segmentation_merged[(x,y)] = min_label.label
                    merged=True
                    has_merged=True
                # updating remaining labels
                for l,line in enumerate(segmentation_merged):
                    for j,value in enumerate(line):
                        if value > max_label.label:
                            segmentation_merged[l][j] -= 1
            if(merged):
                break
        
    print("ending with... {}".format(numpy.amax(segmentation_merged.flatten())))
    #segmentation_merged=merge(labels,image_lab,adjacency)
    
    partition_GCC = dict()
    for node in Gmc.nodes():
        if (node) in Gcc.nodes():
        # assigning cluster as community for every node
            partition_GCC[node] = labels_clustering[dictemb[node]]+(numpy.amax(segmentation)+1)
            #partition_GCC[node] = partition[node]+(label+1)
        else:
            partition_GCC[node] = labels_cc[node-1]'''
            
    #seg_merge, _ = merge(segmentation,image_lab,adjacency)
    #save._savefig(segmentation_merged, image, path_figs+str(i+1)+"_"+filename[:-4]+"_D.png")
    #save._savepreseg(labels, image, path_figs+str(i+1)+"_"+filename[:-4]+"_"+str(number_regions)+"_PRESEG.png")
    #print("PRI LABELS",probabilistic_rand_index(gt_segmentation,labels),number_regions)