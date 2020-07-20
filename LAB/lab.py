# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, img_as_ubyte
from skimage.future import graph
from skimage import data,io,segmentation,color,filters,measure,util
from skimage.metrics import structural_similarity as ssim

from sklearn.preprocessing import normalize
from math import exp, sqrt, ceil
from os import walk
from statistics import mean

import matplotlib.pyplot as plt
import argparse,numpy
import community as community_louvain
import networkx as nx
import cv2

def _filter_edges(edges,d):
    print([e for e in edges if e[2] == d])

def _distance_r_graph(G,R,image=None,distances=None,width=None,height=None):
    # if distances==None then use distance between pixel
    # new empty graph
    Gr = nx.Graph()
    edges = set()
    maxdelta=0
    print("starting from graph with {} vertices and {} edges".format(len(G),len(G.edges())))
    # for every vertex of G
    if distances is None:
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
                    edges.add((u,v,delta))
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
        return Gr
    else:
        for u in G.nodes():
            # we get its distance-R induced subgraph
            Ir = nx.ego_graph(G,u,R)
            # we then add an edge between u and every vertex in Ir, with proper weight
            for v in Ir.nodes():
                if(u != v) and not(distances[u][v]==0):
                    edges.add((u,v,distances[u][v]))
        Gr.add_weighted_edges_from(edges)
        return Gr


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = False, help = "Path to the image")
    ap.add_argument("-p", "--path", required = False, help = "Path to folder")
    args = vars(ap.parse_args())

    # load the image and convert it to a floating point data type
    for (dirpath, dirnames, filenames) in walk(args['path']):
        for filename in filenames:
            image = io.imread(args['path']+"/"+filename)
            image = img_as_float(image)
            image_lab = color.rgb2lab(image)
            #image_lab = (color.rgb2lab(image) + [0,128,128]) // [1,1,1]
            #print("===== IMAGE LAB =====\n {}".format(image_lab))

            # loop over the number of segments
            # apply SLIC and extract (approximately) the supplied number of segments
            numSegments = 250
            # labels with 0 are ignored, seems legit? --- ENFORCE STARTING AT 1, START_LABEL IS FUCKING DUMP
            labels = 1+slic(image, n_segments = numSegments, compactness=50, convert2lab=True, start_label=0)
            print("number of regions from SLIC: {}".format(numpy.amax(labels)))
            print(numpy.amin(labels),numpy.amax(labels))
            regions = measure.regionprops(labels)

            fig = plt.figure("SLIC")
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mark_boundaries(image,labels))

            # computing masks to apply to region histograms
            # loop over the unique segment values
            print("computing means")
            mean_lab = []
            for index,region in enumerate(regions):
                # getting coordinates of region
                coords = region.coords
                cy, cx = region.centroid
                plt.plot(cx, cy, 'ro')
                mean_L, mean_a, mean_b=[],[],[]
                for (x,y) in coords:
                    L,a,b=image_lab[(x,y)]
                    mean_L.append(L)
                    mean_a.append(a)
                    mean_b.append(b)
                mean_lab.append((mean(mean_L),mean(mean_a),mean(mean_b)))
            print("done.")

            # computing adjacency matrix --- need to refine to distance-limited RAG
            adjacency = numpy.zeros((len(numpy.unique(labels))+1, len(numpy.unique(labels))+1))
            for i in range(1,len(adjacency)):
                for j in range(i+1,len(adjacency)):
                    dL=(mean_lab[i-1][0]-mean_lab[j-1][0])**2
                    da=(mean_lab[i-1][1]-mean_lab[j-1][1])**2
                    db=(mean_lab[i-1][2]-mean_lab[j-1][2])**2
                    sim = sqrt(dL+da+db)
                    if(sim<=15):
                        adjacency[i][j] = sim
                        adjacency[j][i] = sim

            # normalizing weights
            # connectivity 2 means 8 neighbors
            G = graph.RAG(labels,connectivity=2)
            labels_pixels = numpy.asarray([[(j*image.shape[1])+i for i in range(1,image.shape[1]+1)] for j in range(image.shape[0])])
            print("computing ALL GRAPH...")
            Gall = graph.RAG(labels_pixels,connectivity=2)
            Gallr = _distance_r_graph(Gall,2,image=image_lab,width=image.shape[1],height=image.shape[0])
            print("done.")
            nx.write_weighted_edgelist(Gallr, 'graph.wgt')
            print("computing louvain")
            partition_allr = community_louvain.best_partition(Gallr)
            print("louvain computed {} communities on {} vertices and {} edges graph".format(max(partition_allr.values())+1,len(Gallr),len(Gallr.edges())))
            labels_from_communities_allr = numpy.zeros((labels_pixels.shape[0],labels.shape[1]),dtype=int)
            for i,line in enumerate(labels_pixels):
                for j,value in enumerate(line):
                    labels_from_communities_allr[i][j] = partition_allr[value]

            # show the output of SLIC+Louvain
            fig = plt.figure("segmentation after community detection")
            ax = fig.add_subplot(1,1,1)
            colored_regions = color.label2rgb(labels_from_communities_allr, image, bg_label=0)
            ax.imshow(mark_boundaries(colored_regions, labels_from_communities_allr, color=(0,0,0), mode='thick'))
            plt.axis("off")
            plt.show()
            print("done.")

            # computing distance-R graph 
            print("computing from adjacency graph")
            Gr = _distance_r_graph(G,8,distances=adjacency)
            print("done")
            #_filter_edges(Gr.edges(data='weight'),0)
            Gd=nx.Graph(Gr)
            plt.show()
            nx.draw(Gd, pos=nx.spring_layout(Gd))
            gimg = color.rgb2gray(image)

            edges = filters.sobel(gimg)
            edges_rgb = color.gray2rgb(edges)

            g = graph.rag_boundary(labels, edges, connectivity=2)
            lc = graph.show_rag(labels, g, edges_rgb, img_cmap=None, edge_cmap='viridis',
                                edge_width=1.2)

            plt.colorbar(lc, fraction=0.03)

            #for u,v in G.edges():
            #    G[u][v]['weight'] = adjacency[u][v]

            partition = community_louvain.best_partition(Gr)
            print("louvain computed {} communities on {} vertices and {} edges graph".format(max(partition.values())+1,len(Gr),len(Gr.edges())))
            labels_from_communities = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
            for i,line in enumerate(labels):
                for j,value in enumerate(line):
                    labels_from_communities[i][j] = partition[value]

            # show the output of SLIC+Louvain
            fig = plt.figure("segmentation after community detection")
            ax = fig.add_subplot(1,1,1)
            colored_regions = color.label2rgb(labels_from_communities, image, bg_label=0)
            ax.imshow(mark_boundaries(colored_regions, labels_from_communities, color=(0,0,0), mode='thick'))
            plt.axis("off")
            plt.show()

            '''edge_map = filters.sobel(color.rgb2gray(image))
            fig = plt.figure("Superpixels -- %d segments" % (numSegments))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mark_boundaries(image, labels))
            plt.axis("off")

            # subclass of networkx 
            RAG_merge = graph.rag_mean_color(image,labels,connectivity=2)
            # should give a clique?
            RAG = graph.rag_boundary(labels,edge_map,labels.ndim)
            dendogram = community_louvain.generate_dendrogram(RAG_merge)
            partition = community_louvain.partition_at_level(dendogram,1)
            # computing labels from communities and regions
            # communities and number are hopefully in one-to-one correspondence
            labels_from_merge = graph.merge_hierarchical(labels,RAG_merge,thresh=35,rag_copy=False,in_place_merge=True,merge_func=merge_mean_color,weight_func=_weight_mean_color)
            fig = plt.figure("segmentation after hierarchical merge")
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mark_boundaries(image, labels_from_merge))
            plt.axis("off")
            labels_from_communities = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
            for i,line in enumerate(labels):
                for j,value in enumerate(line):
                    labels_from_communities[i][j] = partition[value]

            # show the output of SLIC+Louvain
            fig = plt.figure("segmentation after community detection")
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mark_boundaries(image, labels_from_communities))
            plt.axis("off")
            plt.show()'''


