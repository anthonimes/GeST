# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float, img_as_ubyte
from skimage.future import graph
from skimage import data,io,color,filters,measure,util
from skimage.metrics import (adapted_rand_error,
                              variation_of_information)

from sklearn import cluster as cl
from sklearn import metrics
from sklearn.metrics import silhouette_score

from node2vec import Node2Vec

from nfm2vec.src.nfm2vec.nfm import get_nfm_embeddings as nfm 
from utils.parse_matlab import get_groundtruth

from karateclub.node_embedding.neighbourhood import Walklets

from math import exp, sqrt, ceil
from os import walk
from argparse import Namespace
from statistics import mean

import matplotlib.pyplot as plt
import argparse,numpy
import networkx as nx
import json
import warnings, sys
warnings.filterwarnings("ignore")

# function returns WSS score for k values from 1 to kmax
def WSS(points, kmax):
    ssd = []
    print("retrieving best number of clusters...")
    for k in range(1, kmax+1):
        #scaler = StandardScaler()
        #data = scaler.fit_transform(points)
        km = cl.KMeans(n_clusters=k)
        km = km.fit(points)
        ssd.append(km.inertia_)
    print("done.")
    return ssd

def elbow(points):
    from kneed import KneeLocator

    x = range(1,len(points)+1)
    kn = KneeLocator(x, points, curve='convex', direction='decreasing')
    print("best number of clusters: {}".format(kn.knee))

    fig = plt.figure("ELBOW METHOD")
    plt.xlabel('number of clusters k')
    plt.ylabel('Sum of squared distances')
    plt.plot(x, points, 'bx-')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')

    return kn.knee

def hope(dim):
   return  HOPE(dimensions=dim)

def deepwalk(dim):
    return DeepWalk(dimensions=dim)

def walklets(dim):
    return Walklets(dimensions=dim)

def le(dim):
    return LaplacianEigenmaps(dimensions=dim)

def diff2vec(dim):
    return Diff2Vec(dimensions=dim)

def grarep(dim):
    return GraRep(dimensions=dim)

def _apply_k_means(NUM_CLUSTER, embeddings,image,labels,method="",groundtruth=None):

    #for NUM_CLUSTER in NUM_CLUSTERS:
    kmeans = cl.KMeans(n_clusters=NUM_CLUSTER)
    #scaler = StandardScaler()
    #data = scaler.fit_transform(embeddings)
    # removing the line corresponding to dummy vertex
    kmeans.fit(embeddings)

    labels_kmeans = kmeans.labels_
    labels_from_communities_kmeans = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
    for i,line in enumerate(labels):
        for j,value in enumerate(line):
            labels_from_communities_kmeans[i][j] = labels_kmeans[value-1]

    a,b,c,d,e=metrics(groundtruth,labels_from_communities_kmeans,False)

    # show the output of SLIC+Louvain
    '''fig = plt.figure("segmentation after "+method+" "+str(NUM_CLUSTER)+"-means")
    ax = fig.add_subplot(1,1,1)
    colored_regions_kmeans = color.label2rgb(labels_from_communities_kmeans, image, kind='overlay')#,colors=[(1,1,1)])
    #ax.imshow(mark_boundaries(image, labels_from_communities_kmeans, color=(0,0,0), mode='thick'))
    ax.imshow(mark_boundaries(colored_regions_kmeans, labels_from_communities_kmeans, color=(1,1,1), mode='thick'))
    plt.axis("off")'''

    #plt.show()
    return a,b,c,d,e,labels_from_communities_kmeans


def parse_oslom(cluster):
    return [e['id'] for e in cluster['nodes']] 

def _filter_edges(edges,d):
    print([e for e in edges if e[2] == d])

def _distance_r_graph(G,R,image=None,means=None,width=None,height=None,threshold=15):
    # if distances==None then use distance between pixel
    # new empty graph
    Gr = nx.Graph(G)
    # removing edges below threshold
    maxdelta=0
    pairs=0
    edges = set()
    print("starting from graph with {} vertices and {} edges".format(len(G),len(G.edges())))
    # for every vertex of G
    if means is None:
        for i,u in enumerate(G.nodes()):
            # we get its distance-R induced subgraph
            Ir = nx.ego_graph(G,u,R)
            # we then add an edge between u and every vertex in Ir, with proper weight
            for v in Ir.nodes():
                if(u != v):
                    xu=(u-1)//width
                    yu=((u-1)-((xu)*width))
                    xv=(v-1)//width
                    yv=((v-1)-((xv)*width))
                    # FIXME: need to normalize
                    delta=color.deltaE_cie76(image[xu,yu],image[xv,yv])
                    if (u>v):
                        u,v=v,u
                    edges.add((u-1,v-1,delta))
                    if(delta>maxdelta):
                        maxdelta=delta
            if(i%10000 ==0):
                print("10%")
        # normalizing
        edges=list(edges)
        # the highest value means same pixel
        edges = [(e[0],e[1],1-(e[2]/maxdelta)) for e in edges]
        Gr.add_weighted_edges_from(edges)
        return edges,Gr
    else:
        '''for u,v in G.edges():
            dL=(mean_lab[u-1][0]-mean_lab[v-1][0])**2
            da=(mean_lab[u-1][1]-mean_lab[v-1][1])**2
            db=(mean_lab[u-1][2]-mean_lab[v-1][2])**2
            sim = sqrt(dL+da+db)
            #Gr[u][v]['weight']=sim
            #Gr.add_edge(u,v,weight=sim)
            edges.add((u,v,sim))
            #if(sim<=threshold):
            #    Gr[u][v]['weight']=sim
            #else:
            #    Gr.remove_edge(u,v)
        for u in G.nodes():
            # we get its distance-R induced subgraph
            #Ir = nx.ego_graph(G,u,R)
            Ir = list(nx.single_source_shortest_path_length(G ,source=u, cutoff=R).keys())
            #print(Ir)
            # we then add an edge between u and every vertex in Ir, with proper weight
            #for v in Ir.nodes():
            for v in Ir:
                if(u != v):
                    dL=(mean_lab[u-1][0]-mean_lab[v-1][0])**2
                    da=(mean_lab[u-1][1]-mean_lab[v-1][1])**2
                    db=(mean_lab[u-1][2]-mean_lab[v-1][2])**2
                    sim = sqrt(dL+da+db)
                    #if (sim == 0):
                    #    print(u,v)
                    # FIXME: do we need HIGH or SMALL weights?!
                    #if(sim<=threshold and sim > 0):
                        #Gr.add_edge(u,v,weight=sim)
                        # FIXME: WHAT IS THE IMPACT OF WEIGHT 1 ON WALKLETS??
                        #edges.add((u-1,v-1,1.))
                        # HOW THE FUCK CAN 0 HAPPEN?
                    #Gr.add_edge(u,v,weight=sim)
                    edges.add((u,v,sim))
                        # removing useless edges --- UGLY NEED TO BE FIXED
                    #elif Gr.has_edge(u,v):
                    #    Gr.remove_edge(u,v)

        # normalizing
        maxsim=max(edges, key=lambda x: x[2])[2]
        #edges=list(edges)
        #edges=[(e[0],e[1],e[2]/maxsim) for e in edges]
        #edges=[(e[0],e[1],exp(-1./e[2])) for e in edges]
        #edges=[e for e in edges if e[2] != 0 and e[2] != {}]
        edges=[e for e in edges if e[2]<=threshold]
        Gc=nx.Graph()
        Gc.add_weighted_edges_from(edges)
        #for u,v in Gr.edges():
        #    Gr[u][v]['weight']/=maxsim
        #    #Gr[u][v]['weight']=1-Gr[u][v]['weight']'
        #to_remove=[(e[0],e[1]) for e in Gr.edges(data='weight')if e[2] > threshold]
        #Gr.remove_edges_from(to_remove)
        return edges,Gc'''
        for u,v in Gr.edges():
            dL=(mean_lab[u-1][0]-mean_lab[v-1][0])**2
            da=(mean_lab[u-1][1]-mean_lab[v-1][1])**2
            db=(mean_lab[u-1][2]-mean_lab[v-1][2])**2
            sim = sqrt(dL+da+db)
            if(sim<=threshold):
                Gr[u][v]['weight']=sim
            else:
                Gr.remove_edge(u,v)
        for u in G.nodes():
            # we get its distance-R induced subgraph
            #Ir = nx.ego_graph(G,u,R)
            Ir = list(nx.single_source_shortest_path_length(G ,source=u, cutoff=R).keys())
            #print(Ir)
            # we then add an edge between u and every vertex in Ir, with proper weight
            #for v in Ir.nodes():
            for v in Ir:
                if(u != v):
                    dL=(mean_lab[u-1][0]-mean_lab[v-1][0])**2
                    da=(mean_lab[u-1][1]-mean_lab[v-1][1])**2
                    db=(mean_lab[u-1][2]-mean_lab[v-1][2])**2
                    sim = sqrt(dL+da+db)
                    #if (sim == 0):
                    #    print(u,v)
                    # FIXME: do we need HIGH or SMALL weights?!
                    if(sim<=threshold and sim > 0):
                        #Gr.add_edge(u,v,weight=sim)
                        # FIXME: WHAT IS THE IMPACT OF WEIGHT 1 ON WALKLETS??
                        #edges.add((u-1,v-1,1.))
                        # HOW THE FUCK CAN 0 HAPPEN?
                        edges.add((u,v,sim))
                    # removing useless edges --- UGLY NEED TO BE FIXED
                    elif Gr.has_edge(u,v):
                        Gr.remove_edge(u,v)

        # normalizing
        maxsim=max(edges, key=lambda x: x[2])[2]
        #edges=list(edges)
        #edges=[(e[0],e[1],e[2]/maxsim) for e in edges]
        #edges=[(e[0],e[1],exp(-1./e[2])) for e in edges]
        #edges=[e for e in edges if e[2] != 0 and e[2] != {}]
        Gr.add_weighted_edges_from(edges)
        print("ending with graph with {} vertices and {} edges".format(len(Gr),len(Gr.edges())))
        return edges,Gr

def metrics(im_true,im_test,verbose=False):
    error, precision, recall = adapted_rand_error(im_true, im_test)
    splits, merges = variation_of_information(im_true, im_test)
    if(verbose):
        print(f"Adapted Rand error: {error}")
        '''print(f"Adapted Rand precision: {precision}")
        print(f"Adapted Rand recall: {recall}")
        print(f"False Splits: {splits}")
        print(f"False Merges: {merges}")'''
    return error, precision, recall, splits, merges


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

    gt_test = path_groundtruth+"test/"
    gt_train = path_groundtruth+"train/"
    gt_val = path_groundtruth+"val/"

    NUM_CLUSTERS = list(range(2,30))

    # load the image and convert it to a floating point data type
    for (dirpath, dirnames, filenames) in walk(path_val):
        for filename in filenames:
            print("IMAGE:",filename)
            image = io.imread(dirpath+filename)
            image = img_as_float(image)
            image_lab = color.rgb2lab(image)
            image_lab = (color.rgb2lab(image) + [0,128,128]) // [1,1,1]
            #print("===== IMAGE LAB =====\n {}".format(image_lab))

            # loop over the number of segments
            # apply SLIC and extract (approximately) the supplied number of segments
            numSegments = 350
            # labels with 0 are ignored, seems legit? --- ENFORCE STARTING AT 1, START_LABEL IS FUCKING DUMP
            labels = 1+slic(image, n_segments = numSegments, compactness=25, convert2lab=True, start_label=0)
            print("number of regions from SLIC: {}".format(numpy.amax(labels)))

            fig = plt.figure("SLIC")
            ax = fig.add_subplot(1,1,1)
            ax.imshow(mark_boundaries(image,labels))
            #plt.show()

            regions = measure.regionprops(labels)
            # computing masks to apply to region histograms
            # loop over the unique segment values
            print("computing means Lab values")
            mean_lab = []
            for index,region in enumerate(regions):
                # getting coordinates of region
                coords = region.coords
                cy, cx = region.centroid
                #plt.plot(cx, cy, 'ro')
                #plt.show()
                mean_L, mean_a, mean_b=[],[],[]
                for (x,y) in coords:
                    L,a,b=image_lab[(x,y)]
                    mean_L.append(L)
                    mean_a.append(a)
                    mean_b.append(b)
                mean_lab.append((mean(mean_L),mean(mean_a),mean(mean_b)))
            print("done.")

            gt_boundaries, gt_segmentation = get_groundtruth(gt_val+filename.split(".")[0])
            for i in range(5):
                fig = plt.figure("groundtruth segmentation")
                ax = fig.add_subplot(1,1,1)
                colored_regions = color.label2rgb(gt_segmentation[i], image, bg_label=0)
                ax.imshow(mark_boundaries(colored_regions, gt_segmentation[i], color=(1,1,1), mode='thick'))
                plt.axis("off")
                plt.savefig("groundtruth/segmentation/"+filename.split(".")[0]+"_"+str(i+1)+".png")

            '''number_of_segments = []
            # connectivity 2 means 8 neighbors
            G = graph.RAG(labels,connectivity=1)
            #G.add_node(0)

            # computing distance-R graph 
            distance=12
            edges,Gr = _distance_r_graph(G,distance,means=mean_lab,threshold=15)
            pairs=(len(Gr)*(len(Gr)-1))/2
            print("generated {} vertices and {} edges graph ({} pairs)".format(len(Gr),len(Gr.edges()),pairs))
            nx.write_weighted_edgelist(Gr, "graphs/"+filename.split(".")[0]+".wgt", delimiter='\t')
            #nx.write_gpickle(Gr, filename.split(".")[0]+".pkl")

            # FIXME: should be done everytime
            node2vec = Node2Vec(Gr, dimensions=32, walk_length=20, num_walks=10, workers=1)  # Use temp_folder for big graphs
            model = node2vec.fit(window=5, min_count=1)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
            embn2v = []
            for node in Gr.nodes():
                embn2v.append(model.wv.get_vector(str(node)))
            for i in range(5):
                number_of_segments.append(numpy.amax(gt_segmentation[i]))

            for i in [number_of_segments.index(min(number_of_segments)), number_of_segments.index(max(number_of_segments))]:
                print("SEGMENTATION NUMBER: {}".format(i+1))
                print("number of segments: {}".format(numpy.amax(gt_segmentation[i])))
                fig = plt.figure("groundtruth boundaries")
                ax = fig.add_subplot(1,1,1)
                ax.imshow(mark_boundaries(image, gt_boundaries[i], color=(0,0,0), mode='thick'))
                fig = plt.figure("groundtruth segmentation")
                ax = fig.add_subplot(1,1,1)
                colored_regions_gt = color.label2rgb(gt_segmentation[i], image, alpha=1, kind='overlay')#,colors=[(1,1,1)])
                ax.imshow(mark_boundaries(colored_regions_gt, gt_segmentation[i], color=(0,0,0), mode='thick'))
                plt.axis("off")

                labels_felzenszwalb = felzenszwalb(image,scale=300)
                a,b,c,d,e=metrics(gt_segmentation[i],labels_felzenszwalb,True)
                colored_regions = color.label2rgb(labels_felzenszwalb, image, bg_label=0)
                fig = plt.figure("felzenszwalb")
                ax = fig.add_subplot(1,1,1)
                plt.xlabel(str(a))
                ax.imshow(mark_boundaries(colored_regions, labels_felzenszwalb, color=(1,1,1), mode='thick'))
                plt.savefig(filename.split(".")[0]+"_felzenswalb_"+"_"+str(number_of_segments[i])+"_"+str(i+1)+".png")

                # computing adjacency matrix --- need to refine to distance-limited RAG
                adjacency = numpy.zeros((len(numpy.unique(labels))+1, len(numpy.unique(labels))+1))
                pairs=0
                total_pairs=0
                for i in range(1,len(adjacency)):
                    for j in range(i+1,len(adjacency)):
                        dL=(mean_lab[i-1][0]-mean_lab[j-1][0])**2
                        da=(mean_lab[i-1][1]-mean_lab[j-1][1])**2
                        db=(mean_lab[i-1][2]-mean_lab[j-1][2])**2
                        sim = sqrt(dL+da+db)
                        # NEED TO FIND THE RIGHT PARAMETER
                        total_pairs+=1  
                        if(sim<=15):
                            pairs+=1
                            adjacency[i][j] = sim
                            adjacency[j][i] = sim

                print("{} pairs preserved over {}".format(pairs,total_pairs))
                # Node2Vec

                are, prec, rec, splits, merges, segs = [], [], [], [], [], []
                for NUM_CLUSTER in NUM_CLUSTERS:
                    a,b,c,d,e,f=_apply_k_means(NUM_CLUSTER,numpy.asarray(embn2v),image,labels,method="Node2Vec",groundtruth=gt_segmentation[i])
                    are.append(a)
                    prec.append(b)
                    rec.append(c)
                    splits.append(d)
                    merges.append(e)
                    segs.append(f)
                print("Adapted rand error: {}, segmentation {}".format(min(are),are.index(min(are))+2))
                fig = plt.figure("segmentation after Node2Vec "+str(are.index(min(are))+2)+"-means")
                ax = fig.add_subplot(1,1,1)
                colored_regions_kmeans = color.label2rgb(segs[are.index(min(are))], image, kind='overlay')#,colors=[(1,1,1)])
                #ax.imshow(mark_boundaries(image, labels_from_communities_kmeans, color=(0,0,0), mode='thick'))
                ax.imshow(mark_boundaries(colored_regions_kmeans, segs[are.index(min(are))], color=(1,1,1), mode='thick'))
                plt.xlabel(str(min(are))+" ("+str(are.index(min(are))+2)+"-means)")
                #plt.axis("off")
                plt.savefig(filename.split(".")[0]+"_"+str(number_of_segments[i])+"_"+str(i+1)+".png")'''

            # trying with Walkets
            '''args=Namespace()
            args.P=1.0
            args.Q=1.0
            args.dimensions=32
            args.window_size=4
            args.walk_length=80
            args.walk_number=5
            args.input=None
            args.output=None
            args.walk_type="second"
            args.workers=4
            args.min_count=1

            wm = WalkletMachine(args,Gr)
           
            for NUM_CLUSTER in NUM_CLUSTERS:
                a,b,c,d,e=_apply_k_means(NUM_CLUSTER,wm.embedding,image,labels,method="Walklets",groundtruth=gt_segmentation[i])'''
