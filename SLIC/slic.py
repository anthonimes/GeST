# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.future import graph
from skimage import data,io,segmentation,color,filters,measure,util
from skimage.metrics import structural_similarity as ssim

from sklearn.preprocessing import normalize
from math import exp 
from os import walk

import matplotlib.pyplot as plt
import argparse,numpy
import community as community_louvain
import networkx as nx
import cv2

def regions(labels):
    R = []
    for i in range(numpy.amax(labels)+1):
        R.append([])

    for x,line in enumerate(labels):
        for y,region in enumerate(line):
            R[region].append((x,y))
    return R

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = False, help = "Path to folder")
    args = vars(ap.parse_args())

    # load the image and convert it to a floating point data type
    for (dirpath, dirnames, filenames) in walk(args['path']):
        for filename in filenames:
            image = io.imread(args['path']+"/"+filename)
            image = img_as_float(image)

            '''img=data.astronaut()
            labels = segmentation.slic(img, compactness=30, n_segments=400, start_label=1)
            g = graph.rag_mean_color(img, labels)

            labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False,
                                               in_place_merge=True,
                                               merge_func=merge_mean_color,
                                               weight_func=_weight_mean_color)

            out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
            out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
            io.imshow(out)'''
            #io.show()

            # loop over the number of segments
            # apply SLIC and extract (approximately) the supplied number of segments
            numSegments = 250
            labels = 1+slic(image, n_segments = numSegments, compactness=5, start_label=0)
            print(numpy.amax(labels))
            fig = plt.figure("SLIC")
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mark_boundaries(image,labels))

            # computing masks to apply to region histograms
            # loop over the unique segment values
            mask_regions = []
            for (i, segVal) in enumerate(numpy.unique(labels)):
                # construct a mask for the segment
                mask = numpy.zeros(image.shape[:2], dtype = "uint8")
                mask[labels == segVal] = 255
                mask_regions.append(mask)

            # loop over the unique segment values
            regions = measure.regionprops(labels)
            all_regions = []
            all_histo = []
            image_hist = cv2.imread(dirpath+"/"+filename)
            #cv2.cvtColor(image_hist, cv2.COLOR_BGR2RGB)
            cv2.cvtColor(image_hist, cv2.COLOR_BGR2Lab)
            for (i, segVal) in enumerate(numpy.unique(labels)):
                # FIXME: should be the histogram of EXACTLY the region
                #hist = cv2.calcHist([image_hist], [0, 1, 2], mask_regions[i], [16,16,16], [0, 256, 0, 256, 0, 256])
                hist = cv2.calcHist([image_hist], [0, 1, 2], mask_regions[i], [10,16,16], [0, 100, -127, 128, -127, 128])
                hist = cv2.normalize(hist, hist).flatten()
                all_histo.append(hist)

            # computing adjacency matrix
            adjacency = numpy.zeros((len(numpy.unique(labels)), len(numpy.unique(labels))))
            sum_bhattacharyya=0.
            for i in range(len(adjacency)):
                for j in range(i+1,len(adjacency)):
                    if(i != j):
                        # this is TERRIBLY wrong
                        #sim = cv2.compareHist(all_histo[i], all_histo[j], cv2.HISTCMP_BHATTACHARYYA)
                        sim = cv2.compareHist(all_histo[i], all_histo[j], cv2.HISTCMP_INTERSECT)
                        try:
                            adjacency[i][j] = exp(-1./sim)
                            adjacency[j][i] = exp(-1./sim)
                        except(ZeroDivisionError):
                            pass
                        sum_bhattacharyya+=adjacency[i][j]

            # normalizing weights
            adjacency /= sum_bhattacharyya
            G = nx.Graph(adjacency)
            nx.write_weighted_edgelist(G, 'graph.wgt')
            dendrogram = community_louvain.generate_dendrogram(G)
            partition_dendro = community_louvain.partition_at_level(dendrogram,len(dendrogram)-1)
            partition = community_louvain.best_partition(G)
            print(partition, partition_dendro, sep='\n')
            labels_from_communities = numpy.zeros((labels.shape[0],labels.shape[1]),dtype=int)
            for i,line in enumerate(labels):
                for j,value in enumerate(line):
                    labels_from_communities[i][j] = partition[value-1]

            print(numpy.amax(labels_from_communities)+1)
            # show the output of SLIC+Louvain
            fig = plt.figure("segmentation after community detection")
            ax = fig.add_subplot(1,1,1)
            colored_regions = color.label2rgb(labels_from_communities, image)
            ax.imshow(mark_boundaries(colored_regions, labels_from_communities, color=(1,1,1), mode='thick'))
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

            # computing similarity between regions
            '''similarity = numpy.zeros((len(all_regions),len(all_regions)), dtype=int)
            for i in range(len(all_regions)):
                for j in range(i+1,len(all_regions)):
                    similarity[i][j] = ssim(all_regions[i],all_regions[j])

            print(similarity)'''


