# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float, img_as_ubyte
from skimage.future import graph
from sklearn.metrics import silhouette_score, silhouette_samples
from skimage.metrics import (adapted_rand_error,
                              variation_of_information)
from skimage import io,color,measure
from sklearn import cluster as cl
from sklearn import metrics
from sklearn.metrics import silhouette_score

from sklearn.preprocessing import StandardScaler

from gem.embedding.hope import HOPE
from gem.embedding.lle import LocallyLinearEmbedding
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from nfm2vec.src.nfm2vec.nfm import get_nfm_embeddings as nfm 
from utils.parse_matlab import get_groundtruth, get_BSR
from utils.graph import _distance_r_graph, _get_Lab_adjacency
from utils.save import _savefig, _savelabels, _savelabels_seg
from utils.metrics.pri import probabilistic_rand_index
from utils.node2vec.src import node2vec, main
from utils.embeddings.embeddings import walklets

from gensim.models import Word2Vec

from math import exp, sqrt, ceil
from os import walk
from argparse import Namespace
from statistics import mean

import argparse,numpy
import networkx as nx
import json
import warnings, sys
warnings.filterwarnings("ignore")

def learn_embeddings(walks,dimensions=32,window_size=10,min_count=0,workers=4,iter=1):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''

    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=min_count, sg=1, workers=workers, iter=iter)
    #model.save_word2vec_format(args.output)
    
    return model

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

def elbow(points):
    print("elbow method...")

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
    print("done. \nbest number of clusters: {}".format(kn.knee))

    '''fig = plt.figure("ELBOW METHOD")
    plt.xlabel('number of clusters k')
    plt.ylabel('Sum of squared distances')
    plt.plot(x, scores, 'bx-')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.show()'''

    return kn.knee

def silhouette(points,kmax):
    #print("silhouette method...")
    def SSE():
        sse=[]
        for k in range(2, kmax+1):
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

            km = cl.KMeans(n_clusters=k,random_state=10)
            km = km.fit(points)
            labels = km.labels_
            silhouette_avg=silhouette_score(points, labels, metric = 'euclidean')
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(points, labels)
            y_lower = 10
            for i in range(k):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[labels == i]

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

            '''ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(labels.astype(float) / k)
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
            #plt.show()'''

            sse.append(silhouette_avg)
        return sse
        
    scores = SSE()
    x = range(len(scores))
    best = numpy.argmax(scores)+2
    #print("done. \nbest number of clusters: {}".format(best))

    '''fig = plt.figure("SILHOUETTE METHOD")
    plt.xlabel('number of clusters k')
    plt.ylabel('silhouette')
    plt.plot(x, scores, 'bx-')
    plt.vlines(best-2, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.show()'''

    return best

def _apply_clustering(NUM_CLUSTER, embeddings,image,labels,method="",groundtruth=None,GT=None,verbose=False):

    #for NUM_CLUSTER in NUM_CLUSTERS:
    # IMPACT ON AGGLOMERATIVE?
    #scaler = StandardScaler()
    #data = scaler.fit_transform(embeddings)
    data=embeddings
    #clustering = cl.MeanShift(bandwidth=2).fit(embeddings)
    #clustering = cl.KMeans(n_clusters=NUM_CLUSTER,random_state=10).fit(data)
    #clustering.fit(embeddings)
    #clustering = cl.DBSCAN(eps=0.8,metric='euclidean',n_jobs=4).fit(data)
    clustering = cl.AgglomerativeClustering(n_clusters=None,distance_threshold=10).fit(data)

    labels_clustering = clustering.labels_
    labels_from_clustering = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
    for i,line in enumerate(labels):
        for j,value in enumerate(line):
            # +1 needed since 0=background
            labels_from_clustering[i][j] = labels_clustering[value-1]+1

    #error,precision,recall,splits,merges=metrics(groundtruth,labels_from_clustering,verbose)
    pri = probabilistic_rand_index(GT,labels_from_clustering) 

    #return {'error': error, \
    #            'precision': precision, \
    #            'recall': recall, \
    #            'splits': splits, \
    #            'merges': merges}, \
    return            pri, \
                labels_from_clustering, numpy.amax(labels_from_clustering)

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = False, help = "Path to the image")
    ap.add_argument("-p", "--path", required = False, help = "Path to folder")
    argsy = vars(ap.parse_args())
    path_image = argsy['path']+"/images/"
    path_groundtruth = argsy['path']+"/groundTruth/"

    path_test = path_image+"test/"
    path_train = path_image+"train/"
    path_val = path_image+"val/"

    path_results = "results/"
    results_n2v = path_results+"GeST.sc"
    results_BSR= path_results+"BSR.sc"
    results_felz = path_results+"felzenszwalb.sc"
    results_louvain = path_results+"louvain.sc"

    gt_test = path_groundtruth+"test/"
    gt_train = path_groundtruth+"train/"
    gt_val = path_groundtruth+"val/"

    NUM_CLUSTERS = list(range(1,15))

    compactnesses = [ 10, 25, 50, 75 ]
    thresholds = [ 5, 10, 15 ]
    distances = [ 4,5,6 ]
    connectivities = [ 1, 2 ]
    for _compactness in compactnesses:
        for _threshold in thresholds:
            for _distance in distances:
                for _connectivity in connectivities:
                    FELZ_PRI, BSR_PRI, LOUVAIN_PRI, GEST_PRI, HOPE_PRI = [], [], [], [], []
                    # load the image and convert it to a floating point data type
                    for (dirpath, dirnames, filenames) in walk(path_val):
                        for i,filename in enumerate(filenames):
                            if filename.endswith(".jpg"):
                                print("{}: {}".format(i+1,filename))
                                image = io.imread(dirpath+filename)
                                image = img_as_float(image)
                                image_lab = color.rgb2lab(image)
                                image_lab = (color.rgb2lab(image) + [0,128,128]) // [1,1,1]
                                #print("===== IMAGE LAB =====\n {}".format(image_lab))

                                # loop over the number of segments
                                # apply SLIC and extract (approximately) the supplied number of segments
                                numSegments = 200
                                gt_boundaries, gt_segmentation = get_groundtruth(gt_val+filename[:-4]+".mat")
                                # labels with 0 are ignored, seems legit? --- ENFORCE STARTING AT 1, START_LABEL IS FUCKING DUMP
                                labels = 1+slic(image, n_segments = numSegments, compactness=_compactness, convert2lab=True, start_label=0)
                                #print("number of regions from SLIC: {}".format(numpy.amax(labels)))

                                '''_savefig(labels,image,path_results+"SLIC/"+filename[:-4]+".png")            

                                for i in range(5):
                                    _savefig(gt_segmentation[i], image, path_results+"/groundtruth/images/"+filename[:-4]+"_"+str(i+1)+".png",colored=True)
                                    _savelabels(gt_segmentation[i],path_results+"/groundtruth/labels/"+filename[:-4]+"_"+str(i+1)+".lbl")
                                    _savelabels_seg(gt_segmentation[i],path_results+"/groundtruth/segmentation/"+filename[:-4]+"_"+str(i+1)+".seg",filename)'''

                                adjacency=_get_Lab_adjacency(labels,image_lab)

                                number_of_segments = []
                                # connectivity 2 means 8 neighbors
                                G = graph.RAG(labels,connectivity=_connectivity)
                                #G.add_node(0)

                                # computing distance-R graph 
                                distance=_distance
                                Gr = _distance_r_graph(G,distance,adjacency=adjacency,threshold=_threshold)
                                #connected = "connected" if nx.is_connected(Gr) else "not connected"
                                #pairs=(len(Gr)*(len(Gr)-1))/2
                                #print("generated {} vertices and {} edges {} graph ({} pairs)".format(len(Gr),len(Gr.edges()),connected,pairs))
                                #nx.write_weighted_edgelist(Gr, "graphs/"+filename[:-4]+".wgt", delimiter='\t')
                                #nx.write_gpickle(Gr, filename[:-4]+".pkl")'''

                                # FELZENSZWALB
                                #labels_felzenszwalb = felzenszwalb(image,scale=300)+[1]
                                '''_savefig(labels_felzenszwalb, image, path_results+"/felzenszwalb/images/"+filename[:-4]+".png",colored=True)
                                _savelabels(labels_felzenszwalb, path_results+"/felzenszwalb/labels/"+filename[:-4]+".lbl")
                                _savelabels_seg(labels_felzenszwalb,path_results+"/felzenszwalb/segmentation/"+filename[:-4]+".seg",filename)'''

                                # BSR
                                '''labels_BSR = get_BSR(dirpath+"/BSR/"+filename[:-4]+".mat")
                                _savefig(labels_BSR, image, path_results+"/BSR/images/"+filename[:-4]+".png",colored=True)
                                _savelabels(labels_BSR, path_results+"/BSR/labels/"+filename[:-4]+".lbl")
                                _savelabels_seg(labels_BSR,path_results+"/BSR/segmentation/"+filename[:-4]+".seg",filename)

                                # LOUVAIN
                                partition = community_louvain.best_partition(Gr)
                                print("louvain computed {} communities on {} vertices and {} edges graph".format(max(partition.values())+1,len(Gr),len(Gr.edges())))
                                labels_from_communities = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
                                for k,line in enumerate(labels):
                                    for j,value in enumerate(line):
                                        labels_from_communities[k][j] = partition[value]+1
                                _savefig(labels_from_communities, image, path_results+"/louvain/images/"+filename[:-4]+".png",colored=True)
                                _savelabels(labels_from_communities,path_results+"/louvain/labels/"+filename[:-4]+".lbl")
                                _savelabels_seg(labels_from_communities,path_results+"/louvain/segmentation/"+filename[:-4]+".seg",filename)'''

                                # LLE
                                '''lle = LocallyLinearEmbedding(d=32)
                                Y, t = lle.learn_embedding(graph=Gr, edge_f=None, is_weighted=True, no_python=True)
                                emblle = lle.get_embedding()
                                max_pri = 0
                                for best_k in NUM_CLUSTERS:
                                    _, pri, segmentation=_apply_clustering(best_k,numpy.asarray(emblle),image,labels,method="HOPE",groundtruth=gt_segmentation[i], GT=gt_segmentation)
                                    if(pri > max_pri):
                                        max_pri = pri
                                GEST_PRI.append(max_pri)
                                #GEST_PRI.append(probabilistic_rand_index(gt_segmentation,segmentation))
                                print(max_pri,probabilistic_rand_index(gt_segmentation,labels_felzenszwalb))'''

                                # HOPE
                                '''hope = HOPE(d=32,beta=0.12)
                                Y, t = hope.learn_embedding(graph=Gr, edge_f=None, is_weighted=True, no_python=True)
                                embhope = hope.get_embedding()
                                best_k=0
                                #for best_k in NUM_CLUSTERS:
                                _, pri, segmentation=_apply_clustering(best_k,numpy.asarray(embhope),image,labels,method="HOPE",groundtruth=gt_segmentation[i], GT=gt_segmentation)
                                    #if(pri > max_pri):
                                    #    max_pri = pri
                                HOPE_PRI.append(pri)
                                #GEST_PRI.append(probabilistic_rand_index(gt_segmentation,segmentation))
                                print(mean(HOPE_PRI),pri,probabilistic_rand_index(gt_segmentation,labels_felzenszwalb))'''

                                # Node2Vec
                                n2v = node2vec.Graph(Gr,False,1,1)
                                n2v.preprocess_transition_probs()
                                walks = n2v.simulate_walks(80,40)
                                model=learn_embeddings(walks)
                                embn2v=[]
                                for node in Gr.nodes():
                                    embn2v.append(model.wv.get_vector(str(node)).tolist())

                                '''node2vec = Node2Vec(Gr, dimensions=32, walk_length=40, workers=4, p=1, q=0.2, quiet=True)  # Use temp_folder for big graphs
                                model = node2vec.fit(window=5, min_count=1)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
                                for node in Gr.nodes():
                                    embn2v.append(model.wv.get_vector(str(node)).tolist())
                                for i in range(5):
                                    number_of_segments.append(numpy.amax(gt_segmentation[i]))'''
                                max_pri = 0
                                max_nb_clusters=0
                                best_k=0
                                #best_k = elbow(embn2v,20)
                                #best_k = silhouette(numpy.asarray(embn2v),20)
                                #for best_k in NUM_CLUSTERS:
                                pri, segmentation,nb_clusters=_apply_clustering(best_k,numpy.asarray(embn2v),image,labels,method="Node2Vec",groundtruth=gt_segmentation[0], GT=gt_segmentation)
                                #    if(pri > max_pri):
                                #        max_pri = pri
                                #        max_nb_clusters = nb_clusters
                                GEST_PRI.append(pri)
                                #felz_pri=probabilistic_rand_index(gt_segmentation,labels_felzenszwalb)
                                #FELZ_PRI.append(felz_pri)
                                #print(max_nb_clusters,mean(GEST_PRI),max_pri)
                                '''print("========== GeST ===========\n")
                                #print("{}\t{}\t{}\t{}\t{}\t{}".format(scores['error'],scores['precision'],scores['recall'],scores['splits'],scores['merges'],probabilistic_rand_index(gt_segmentation,segmentation)))

                                #for i in [number_of_segments.index(min(number_of_segments)), number_of_segments.index(max(number_of_segments))]:
                                #for i in range(5):'''
                                # Felzenszwalb
                                #error,precision,recall,splits,merges=metrics(gt_segmentation[i],labels_felzenszwalb,False)
                                #print("========== FELZENSZWALB ===========\n")
                                #FELZ_PRI.append(probabilistic_rand_index(gt_segmentation,labels_felzenszwalb))
                                #print("{}\t{}\t{}\t{}\t{}\t{}".format(error,precision,recall,splits,merges,probabilistic_rand_index(gt_segmentation,labels_felzenszwalb)))
                                '''with open(results_felz, "a") as f:
                                    f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"\
                                    .format(filename,i+1,number_of_segments[i],numpy.amax(labels_felzenszwalb),error,precision,recall,splits,merges))'''

                                # BSR
                                #error,precision,recall,splits,merges=metrics(gt_segmentation[i],labels_BSR,False)
                                #print("========== BSR ===========\n")
                                #BSR_PRI.append(probabilistic_rand_index(gt_segmentation,labels_BSR))
                                #print("{}\t{}\t{}\t{}\t{}\t{}".format(error,precision,recall,splits,merges,probabilistic_rand_index(gt_segmentation,labels_BSR)))
                                '''with open(results_BSR, "a") as f:
                                    f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"\
                                    .format(filename,i+1,number_of_segments[i],numpy.amax(labels_BSR),error,precision,recall,splits,merges))'''

                                # Louvain
                                '''error,precision,recall,splits,merges=metrics(gt_segmentation[i],labels_from_communities,False)
                                print("========== LOUVAIN ===========\n")
                                print(probabilistic_rand_index(gt_segmentation,labels_from_communities))
                                #print("{}\t{}\t{}\t{}\t{}\t{}".format(error,precision,recall,splits,merges,probabilistic_rand_index(gt_segmentation,labels_from_communities)))
                                with open(results_louvain, "a") as f:
                                    f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"\
                                    .format(filename,i+1,number_of_segments[i],numpy.amax(labels_from_communities),error,precision,recall,splits,merges))

                                # Node2Vec
                                with open(results_n2v, "a") as f:
                                    are, prec, rec, splits, merges, segs = [], [], [], [], [], []
                                    for NUM_CLUSTER in NUM_CLUSTERS:
                                        scores, segmentation=_apply_clustering(NUM_CLUSTER,numpy.asarray(embn2v),image,labels,method="Node2Vec",groundtruth=gt_segmentation[i])
                                        are.append(scores['error'])
                                        prec.append(scores['precision'])
                                        rec.append(scores['recall'])
                                        splits.append(scores['splits'])
                                        merges.append(scores['merges'])
                                        segs.append(segmentation)
                                        #print("{}\t{}\t{}\t{}\t{}".format(scores['error'],scores['precision'],scores['recall'],scores['splits'],scores['merges']))
                                    best=are.index(min(are))
                                    f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"\
                                    .format(filename,i+1,number_of_segments[i],numpy.amax(segs[best]),are[best],prec[best],rec[best],splits[best],merges[best]))
                                    #print("{}\t{}\t{}\t{}\t{}\t{}".format(filename,are[best],prec[best],rec[best],splits[best],merges[best]))
                                    #print("Adapted rand error: {}, segmentation {}".format(min(are),are.index(min(are))+2))

                                    _savefig(segs[best], image, path_results+"/GeST/images/"+filename[:-4]+"_"+str(number_of_segments[i])+"_"+str(i+1)+"_"+str(numpy.amax(segs[best]))+".png",colored=True)
                                    _savelabels(segs[best], path_results+"/GeST/labels/"+filename[:-4]+"_"+str(number_of_segments[i])+"_"+str(i+1)+"_"+str(numpy.amax(segs[best]))+".lbl")
                                    _savelabels_seg(segs[best],path_results+"/GeST/segmentation/"+filename[:-4]+"_"+str(number_of_segments[i])+"_"+str(i+1)+"_"+str(numpy.amax(segs[best]))+".seg",filename)
                    print("FELZENSZWALB PRI: {}".format(mean(FELZ_PRI)))
                    print("BSR PRI: {}".format(mean(BSR_PRI)))'''
                    print("compactness: {}\t threshold: {}\t distance: {}\t connectivity: {}\t PRI: {}".format(_compactness,_threshold,_distance,_connectivity,mean(GEST_PRI)))
