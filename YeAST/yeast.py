# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float, img_as_ubyte
from skimage.future import graph
from skimage import data,io,color,filters,measure,util
from skimage.metrics import structural_similarity as ssim

from sklearn import cluster as cl
from sklearn import metrics
from sklearn.preprocessing import normalize

from nfm2vec.src.nfm2vec.nfm import get_nfm_embeddings as nfm 
from utils.walklets.src.walklets import WalkletMachine

from karateclub.node_embedding.neighbourhood import HOPE
from karateclub.node_embedding.neighbourhood import DeepWalk
from karateclub.node_embedding.neighbourhood import Walklets
from karateclub.node_embedding.neighbourhood import GraRep
from karateclub.node_embedding.neighbourhood import LaplacianEigenmaps
from karateclub.node_embedding.neighbourhood import Diff2Vec

from math import exp, sqrt, ceil
from os import walk
from argparse import Namespace
from statistics import mean

import matplotlib.pyplot as plt
import argparse,numpy
import community as community_louvain
import networkx as nx
import cv2
import oslom
import json
import warnings, sys
warnings.filterwarnings("ignore")

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

def _apply_k_means(NUM_CLUSTERS, embeddings,image,labels,method=""):
    fig = plt.figure("SLIC")
    ax = fig.add_subplot(1,1,1)
    ax.imshow(mark_boundaries(image,labels))

    for NUM_CLUSTER in NUM_CLUSTERS:
        kmeans = cl.KMeans(n_clusters=NUM_CLUSTER)
        # removing the line corresponding to dummy vertex
        #kmeans.fit(embeddings[1:])
        kmeans.fit(embeddings)
         
        labels_kmeans = kmeans.labels_

        labels_from_communities_kmeans = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
        for i,line in enumerate(labels):
            for j,value in enumerate(line):
                labels_from_communities_kmeans[i][j] = labels_kmeans[value-1]

        # show the output of SLIC+Louvain
        fig = plt.figure("segmentation after "+method+" "+str(NUM_CLUSTER)+"-means")
        ax = fig.add_subplot(1,1,1)
        colored_regions_kmeans = color.label2rgb(labels_from_communities_kmeans, image, alpha=1, kind='overlay')#,colors=[(1,1,1)])
        #ax.imshow(mark_boundaries(image, labels_from_communities_kmeans, color=(0,0,0), mode='thick'))
        ax.imshow(mark_boundaries(colored_regions_kmeans, labels_from_communities_kmeans, color=(0,0,0), mode='thick'))
        plt.axis("off")

    plt.show()


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
        print("ending with graph with {} vertices and {} edges".format(len(Gr),len(Gr.edges())))
        return edges,Gr
    else:
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
                    if(sim<=threshold):
                        #Gr.add_edge(u,v,weight=sim)
                        # FIXME: WHAT IS THE IMPACT OF WEIGHT 1 ON WALKLETS??
                        edges.add((u-1,v-1,1.))
                        #edges.add((u-1,v-1,sim))
                    # removing useless edges --- UGLY NEED TO BE FIXED
                    elif Gr.has_edge(u-1,v-1):
                        Gr.remove_edge(u-1,v-1)

        # normalizing
        #maxsim=max(edges, key=lambda x: x[2])[2]
        #edges=list(edges)
        #edges=[(e[0],e[1],e[2]/maxsim) for e in edges]
        #edges=[(e[0],e[1],1-e[2]) for e in edges]
        edges=[e for e in edges if e[2] != 0]
        Gr.add_weighted_edges_from(edges)
        print("ending with graph with {} vertices and {} edges".format(len(Gr),len(Gr.edges())))
        return edges,Gr


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = False, help = "Path to the image")
    ap.add_argument("-p", "--path", required = False, help = "Path to folder")
    argsy = vars(ap.parse_args())
    NUM_CLUSTERS = [2,3,5,8]

    # load the image and convert it to a floating point data type
    for (dirpath, dirnames, filenames) in walk(argsy['path']):
        for filename in filenames:
            image = io.imread(argsy['path']+"/"+filename)
            image = img_as_float(image)

            labels_felzenszwalb = felzenszwalb(image,scale=300)
            colored_regions = color.label2rgb(labels_felzenszwalb, image, bg_label=0)
            '''fig = plt.figure("felzenszwalb")
            ax = fig.add_subplot(1,1,1)
            ax.imshow(mark_boundaries(colored_regions, labels_felzenszwalb, color=(1,1,1), mode='thick'))
            plt.show()'''

            image_lab = color.rgb2lab(image)
            image_lab = (color.rgb2lab(image) + [0,128,128]) // [1,1,1]
            #print("===== IMAGE LAB =====\n {}".format(image_lab))

            # loop over the number of segments
            # apply SLIC and extract (approximately) the supplied number of segments
            numSegments = 450
            # labels with 0 are ignored, seems legit? --- ENFORCE STARTING AT 1, START_LABEL IS FUCKING DUMP
            labels = 1+slic(image, n_segments = numSegments, compactness=25, convert2lab=True, start_label=0)
            print("number of regions from SLIC: {}".format(numpy.amax(labels)))

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

            # computing adjacency matrix --- need to refine to distance-limited RAG
            '''adjacency = numpy.zeros((len(numpy.unique(labels))+1, len(numpy.unique(labels))+1))
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

            print("{} pairs preserved over {}".format(pairs,total_pairs))'''
            # connectivity 2 means 8 neighbors
            G = graph.RAG(labels,connectivity=2)
            #G.add_node(0)

            # computing distance-R graph 
            distance=12
            print("computing adjacency graph with distance {}".format(distance))
            edges,Gr = _distance_r_graph(G,distance,means=mean_lab,threshold=15)
            print("generated {} vertices and {} edges graph".format(len(Gr),len(Gr.edges())))
            nx.write_weighted_edgelist(Gr, 'graph.wgt', delimiter='\t')
            print("done")

            methods = {
                "GraRep": grarep,
                "Walklets": walklets,
                "Hope": hope,
                "Deepwalk": deepwalk,
                "Diff2vec": diff2vec,
                "Laplacianeigenmaps": le
            }

            # FIXME: G HAS NODES 1->N, Gr HAS 0->N-1: IS EVERYTHING COHERENT WITH K-MEANS????
            embeddings_liste = [ "Hope", "Deepwalk", "Diff2vec", "Walklets" ]
            method_embeddings = [ (emb, methods[emb]) for emb in embeddings_liste ]

            for idx, (name, method) in enumerate(method_embeddings):
                print(name)
                Gc=nx.Graph(Gr)
                algo=method(32)
                Gc.remove_nodes_from(list(nx.isolates(Gc)))
                algo.fit(graph=Gc)
                embedding= algo.get_embedding()
                _apply_k_means(NUM_CLUSTERS,embedding,image,labels,method=name)

            # trying with Walkets
            '''args=Namespace()
            args.P=1.0
            args.Q=1.0
            args.dimensions=16
            args.window_size=4
            args.walk_length=80
            args.walk_number=5
            args.input=None
            args.output=None
            args.walk_type="second"
            args.workers=4
            args.min_count=1

            wm = WalkletMachine(args,Gr)
            
            _apply_k_means(NUM_CLUSTERS,wm.embedding,image,labels)

            partition = community_louvain.best_partition(Gr)
            print("louvain computed {} communities on {} vertices and {} edges graph".format(max(partition.values())+1,len(Gr),len(Gr.edges())))
            labels_from_communities = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
            for i,line in enumerate(labels):
                for j,value in enumerate(line):
                    labels_from_communities[i][j] = partition[value]

            fig = plt.figure("segmentation after louvain community detection")
            ax = fig.add_subplot(1,1,1)
            colored_regions = color.label2rgb(labels_from_communities, image, bg_label=0)
            ax.imshow(mark_boundaries(colored_regions, labels_from_communities, color=(1,1,1),  mode='thick'))
            plt.axis("off")

            partition_vector=list(partition.values())
            edges = [(edge[0],edge[1],edge[2]) for edge in Gr.edges.data('weight', default=1.)]
            # nodes are from 0 (empty node) to n
            weights = [0]+[Gr.degree(u) for u in Gr.nodes()]
            number_communities = max(partition_vector)+1
            # +1 needed for dummy node 0
            nodes = len(Gr)+1
            # computing embedding vectors --- CSR numpy format: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
            # FIXME : COMPUTE REAL NODE F-MEASURE VECTORS
            np,nr,embedding_matrix = nfm(edges, weights, partition_vector, number_communities, nodes)

            labels_from_communities = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
            for i,line in enumerate(labels):
                for j,value in enumerate(line):
                    labels_from_communities[i][j] = partition[value]

            # computing clusters
            _apply_k_means(NUM_CLUSTERS,embedding_matrix,image,labels)

            # TRYING OSLOM
            argso = Namespace()
            argso.min_cluster_size = 0
            argso.oslom_exec = oslom.DEF_OSLOM_EXEC
            argso.oslom_args = oslom.DEF_OSLOM_ARGS
            argso.oslom_output = "here"
            argso.oslom_output_dir = "OSLOM"
            clusters,logs = oslom.run_in_memory(argso, edges)
            communities = []
            for cluster in clusters['clusters']:
                communities.append(parse_oslom(cluster))
            partition=dict()
            for index,community in enumerate(communities):
                for elt in community:
                    partition[elt]=index
            # affect a particular color to homeless nodes --- CRAPPY
            garbage=max(partition.values())+1
            print("OSLOM computed {} communities on {} vertices and {} edges graph".format(garbage+1,len(Gr),len(Gr.edges())))
            # for consistency with embeddings --- this node does not even exist in the graph... ?!
            partition[0]=garbage
            for u in Gr.nodes():
                if u not in partition:
                    partition[u] = garbage
            partition_vector=list(partition.values())
            edges = [(edge[0],edge[1],edge[2]) for edge in Gr.edges.data('weight', default=1.)]
            # nodes are from 0 (empty node) to n
            weights = [0]+[Gr.degree(u) for u in Gr.nodes()]
            number_communities = max(partition_vector)+1
            # +1 needed for dummy node 0
            nodes = len(Gr)+1
            # computing embedding vectors --- CSR numpy format: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
            np,nr,embedding_matrix = nfm(edges, weights, partition_vector, number_communities, nodes)

            labels_from_communities = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
            for i,line in enumerate(labels):
                for j,value in enumerate(line):
                    labels_from_communities[i][j] = partition[value]

            fig = plt.figure("segmentation after OSLOM community detection")
            ax = fig.add_subplot(1,1,1)
            colored_regions = color.label2rgb(labels_from_communities, image, bg_label=0)
            ax.imshow(mark_boundaries(colored_regions, labels_from_communities, color=(0,0,0), mode='thick'))
            plt.axis("off")

            # computing clusters
            _apply_k_means(NUM_CLUSTERS,embedding_matrix,image,labels)'''
