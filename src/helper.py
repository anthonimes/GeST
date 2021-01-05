from skimage import io,color,measure,img_as_ubyte
from skimage.segmentation import mark_boundaries

from math import sqrt, ceil
import random

import sklearn.metrics
import sklearn.cluster

import numpy
import argparse

def _parse_args():    
    """
    Builds and returns the argument parser for GeST.
    """
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-p", "--path", required=True, help="path to folder containing images")
    ap.add_argument("-m", "--method", required=False, default="msp", help="pre-segmentation method")
    ap.add_argument( "--sigma", required=False, help="kernel parameter", default=125)

    ap.add_argument("-n", "--nclusters", required=False, default=21, help="number of clusters")
    ap.add_argument("--silhouette", required=False, default=False, action="store_true", help="use silhouette method instead of fixed number of clusters")

    ap.add_argument("--hs", required=False, help="spatial radius", default=7)
    ap.add_argument("--hr", required=False, help="range radius", default=4.5)
    ap.add_argument( "--mind", required=False, help="min density", default=50)

    ap.add_argument("--merge", required=False, default=False, action="store_true", help="apply merging procedure")
    ap.add_argument("--contiguous", required=False, default=False, action="store_true", help="compute contiguous regions")
    ap.add_argument("-s", "--save", required=False, default=False, action="store_true", help="save files to hard drive")
    arguments=vars(ap.parse_args())

    arguments['n_cluster']= None if arguments['silhouette'] else int(arguments['nclusters'])
    arguments['sigma']=float(arguments['sigma'])
    
    return arguments

def _colors(segmentation,image):
    """
    Compute mean color of each region of a given segmentation
    
    :param segmentation:
        The segmentation to compute mean from
    :param image:
        The image to get pixels from
    """
    regions=measure.regionprops(segmentation)
    colors=[0]*len(regions)
    for index,region in enumerate(regions):
        # getting coordinates of region
        coords=region.coords
        size_coords=len(coords)
        R_value, G_value, B_value=[0]*size_coords,[0]*size_coords,[0]*size_coords
        for p,(x,y) in enumerate(coords):
            R,G,B=image[(x,y)]
            R_value[p]=R
            G_value[p]=G
            B_value[p]=B
        colors[index]=(sum(R_value)/size_coords,sum(G_value)/size_coords,sum(B_value)/size_coords)
    return colors

def _savepreseg(presegmentation=None,image=None,path=None):
    """
    Write the presegmentation to harddrive as an image with marked boundaries

    :param segmentation:
        The presegmentation to write
    :param image:
        The original image
    :param path:
        The path to write the result to
    """
    io.imsave(path,img_as_ubyte(mark_boundaries(image,segmentation, mode='thick')))

def _savefig(segmentation=None,image=None,path=None):
    """
    Write the presegmentation to harddrive as an image with marked boundaries

    :param segmentation:
        The segmentation to write
    :param image:
        The original image
    :param path:
        The path to write the result to
    """
    colored_regions=color.label2rgb(segmentation, image, alpha=1, colors=_colors(segmentation,image), bg_label=0)
    io.imsave(path,img_as_ubyte(colored_regions))

def _color_features(labels,image_lab):
    """
    Function that computes a feature vector for a given image and a set of segments

    :param labels:
        The segmentation to start from
    :param image_lab:
        The image in L*a*b* space
    """
    regions=measure.regionprops(labels)
    number_regions=len(regions)
    mean_lab, stdev_lab=[0]*number_regions, [0]*number_regions
    feature_vector=[0]*number_regions
    for i,region in enumerate(regions):
        # getting coordinates of region
        coords=region.coords
        size_coords=len(coords)
        L_value, a_value, b_value=[0]*len(coords),[0]*len(coords),[0]*len(coords)
        for (l,(x,y)) in enumerate(coords):
            L,a,b=image_lab[(x,y)]
            L_value[l]=L
            a_value[l]=a
            b_value[l]=b
        mean_lab[i]=[sum(L_value)/size_coords,sum(a_value)/size_coords,sum(b_value)/size_coords]
        stdev_lab[i]=[numpy.std(L_value),numpy.std(a_value),numpy.std(b_value)]
        feature_vector[i]=mean_lab[i]+stdev_lab[i]
    return feature_vector

def silhouette(points,kmax):
    """
    Function computing the number of segments that achieve the best silhouette score
    """
    def SSE():
        sse=[]
        for k in range(2, kmax):
            km=sklearn.cluster.AgglomerativeClustering(n_clusters=k,affinity='cosine',linkage='average',distance_threshold=None).fit(points)
            labels_clustering=km.labels_
            silhouette_avg=sklearn.metrics.silhouette_score(points, labels_clustering, metric='cosine')
            sse.append(silhouette_avg)
        return sse
        
    scores=SSE()
    best=scores.index(max(scores))+2
    return best

def _write(g,arguments):
    """
    Write attributes of a GeST object to hard drive

    :param g:
        The GeST object
    :param arguments:
        The argument parser as returned by _parse_args()
    """
    from networkx import write_gpickle, write_weighted_edgelist
    from numpy import save
    from pickle import dump
    from os import walkdirs
    from src.helper import _savepreseg, _savefig

    _spatial_radius=float(arguments['hs']) #hs
    _range_radius=float(arguments['hr']) #hr
    _min_density=int(arguments['mind']) #mind
    common=arguments['method']+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(arguments['sigma'])+"/"

    path_graphs = "results/graphs/"+common
    path_pickles = "results/pickles/"+common
    path_labels_msp = "results/labels/"+common
    path_labels = "results/labels/"+common
    path_presegs = "results/presegs/"+common
    path_embeddings = "results/embeddings/"+common
    path_clusterings = "results/clusterings/"+common
    path_segmentation = "results/segmentation/"+common

    makedirs(path_graphs,exist_ok=True)
    makedirs(path_pickles,exist_ok=True)
    makedirs(path_labels,exist_ok=True)
    makedirs(path_presegs,exist_ok=True)
    makedirs(path_embeddings,exist_ok=True)
    makedirs(path_clusterings,exist_ok=True)
    makedirs(path_segmentation,exist_ok=True)

    dump(g._presegmentation,open(path_labels+str(i+1)+"_"+filename[:-4]+".preseg","wb"))
    dump(g._segmentation,open(path_labels+str(i+1)+"_"+filename[:-4]+".seg","wb"))
    save(path_embeddings+filename[:-4]+".emb",g._embeddings)
    write_gpickle(g._RAG, path_pickles+str(i+1)+"_"+filename[:-4]+".pkl")
    write_weighted_edgelist(g._RAG, path_graphs+filename[:-4]+".wgt", delimiter='\t')
    _savepreseg(g._presegmentation, g._image, path_presegs+filename[:-4]+".png")
    _savefig(g._segmentation, g._image, path_segmentation+str(i+1)+"_"+filename[:-4]+"_"+str(g._number_of_regions)+".png")

def _display(g, arguments):
    """
    Function that plots:
        - the initial image
        - the presegmented image (colormap)
        - the segmented image (colormap)
        - the segmented image (with mean color of each region)a
        - the merged image (colormap ---if appropriate)
        - the merged image (with mean color of each region ---if appropriate)
    """
    from skimage import color
    from skimage import measure
    from matplotlib import pyplot as plt
    from src.helper import _colors

    if(arguments['merge']):
        fig, ax = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
    else:
        fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    ax[0][0].imshow(g._image)
    ax[0][0].set_title("initial image")
    ax[0][1].imshow(g._presegmentation)
    ax[0][1].set_title("initial segmentation")
        
    colored_regions = color.label2rgb(g._segmentation, g._image, alpha=1, colors=_colors(g._segmentation,g._image), bg_label=0)
    ax[1][0].imshow(g._segmentation)
    ax[1][0].set_title('final segmentation')
    colored_regions = color.label2rgb(g._segmentation, g._image, alpha=1, colors=_colors(g._segmentation,g._image), bg_label=0)
    ax[1][1].imshow(colored_regions)
    ax[1][1].set_title('colored final segmentation')

    if(arguments['merge']):
        ax[2][0].imshow(g._segmentation_merged)
        ax[2][0].set_title('merged segmentation')
        colored_regions = color.label2rgb(g._segmentation_merged, g._image, alpha=1, colors=_colors(g._segmentation_merged,g._image), bg_label=0)
        ax[2][1].imshow(colored_regions)
        ax[2][1].set_title('colored merged segmentation')

        for a in ax.ravel():
            a.set_axis_off()
    
    plt.tight_layout()
    plt.show()
