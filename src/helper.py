from skimage import io,color,measure,img_as_ubyte
from skimage.segmentation import mark_boundaries
from skimage.feature import hog

from math import sqrt, ceil
import random

import sklearn.metrics
import sklearn.cluster

import numpy
import argparse
import matplotlib.pyplot as plt

def _parse_args():    
    """
    Builds and returns the argument parser for GeST.
    """
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-p", "--path", required=True, help="path to folder containing images")
    ap.add_argument("-g", "--groundtruth", required=False, help="path to groundtruth images",default="")
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
    ap.add_argument("-v", "--verbose", required=False, default=False, action="store_true", help="print information about segmentation")
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

# FIXME: merge functions and use a boolean for boundaries
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

# ===== FEATURE FUNCTIONS =====
def _hog_channel_gradient(image,multichannel=False):
    """Compute unnormalized gradient image along `row` and `col` axes.
    Parameters
    ----------
    channel : (M, N) ndarray
        Grayscale image or one of image channel.
    Returns
    -------
    g_row, g_col : channel gradient along `row` and `col` axes correspondingly.
    """
    if(not(multichannel)):
        g_row = numpy.empty(image.shape, dtype=numpy.double)
        g_row[0, :] = 0
        g_row[-1, :] = 0
        g_row[1:-1, :] = image[2:, :] - image[:-2, :]
        g_col = numpy.empty(image.shape, dtype=numpy.double)
        g_col[:, 0] = 0
        g_col[:, -1] = 0
        g_col[:, 1:-1] = image[:, 2:] - image[:, :-2]

    else:
        g_row_by_ch = numpy.empty_like(image, dtype=numpy.double)
        g_col_by_ch = numpy.empty_like(image, dtype=numpy.double)
        magnitude = numpy.empty_like(image, dtype=numpy.double)

        for idx_ch in range(image.shape[2]):
            channel=image[:, :, idx_ch]
            g_row = numpy.empty(channel.shape, dtype=numpy.double)
            g_row[0, :] = 0
            g_row[-1, :] = 0
            g_row[1:-1, :] = channel[2:, :] - channel[:-2, :]
            g_col = numpy.empty(channel.shape, dtype=numpy.double)
            g_col[:, 0] = 0
            g_col[:, -1] = 0
            g_col[:, 1:-1] = channel[:, 2:] - channel[:, :-2]

            g_row_by_ch[:, :, idx_ch], g_col_by_ch[:, :, idx_ch] = \
                g_col, g_row
            magnitude[:, :, idx_ch] = numpy.hypot(g_row_by_ch[:, :, idx_ch],
                                            g_col_by_ch[:, :, idx_ch])

        # For each pixel select the channel with the highest gradient magnitude
        idcs_max = magnitude.argmax(axis=2)
        rr, cc = numpy.meshgrid(numpy.arange(image.shape[0]),
                             numpy.arange(image.shape[1]),
                             indexing='ij',
                             sparse=True)
        g_row = g_row_by_ch[rr, cc, idcs_max]
        g_col = g_col_by_ch[rr, cc, idcs_max]

    # magnitude and direction
    magnitude = numpy.hypot(g_col,g_row)
    orientation = numpy.rad2deg(numpy.arctan2(g_row,g_col)) % 180

    # ---DEV--- 
    # computes this on a RGB (or LAB ?) image instead of gray scale 
    # compute this for every channel, and preserve the max!
    # ---DEV---
    return magnitude, orientation

# ---DEV--- compute 9x1 vector (bins) for a given region
def _get_bins(magnitude, orientation, regions):
#def _get_bins(regions):
    all_bins = list()
    for region in regions:
        bins = [0]*9
        size=0
        for (x,y) in region.coords:
            m,o = magnitude[(x,y)], orientation[(x,y)]
            # orientation defines bin, magnitude defines vote
            # ---DEV--- 
            # trying to implement vote according to magnitude
            # ---DEV---
            to_bin = o//20
            bins[int(to_bin)]+=1
            '''to_current_bin = 1-((o%20)/20) # 0.75
            to_next_bin = 1-to_current_bin # 0.25
            bins[int(to_bin)]+=m*(to_current_bin) # 85*0.75 = 63.75 to bin 8
            bins[int(to_bin+1)%9]+=m*(to_next_bin) # 85*0.25 = 21.25 to bin 8+1 % 9 = 0
            size+=m'''
        # we provide the percentage of bins
        #all_bins.append([(bins[i],size) for i in range(9)])
        all_bins.append([(bins[i],len(region.coords)) for i in range(9)])
    return all_bins

# FIXME: is there something built-in in skimage?
def _hog_feature(labels,intensity_image,orientation,magnitude):
    """
    Function that computes a feature vector for a given image and a set of segments

    :param labels:
        The segmentation to start from
    :param image_lab:
        The image in L*a*b* space
    """
    regions=measure.regionprops(labels,intensity_image=intensity_image)
    feature_vector=[0]*len(regions)

    orientation, magnitude = _hog_channel_gradient(intensity_image)
    # ---DEV--- is it better to compute magnitude and orientation on each region image separately?
    for i,region in enumerate(regions):
        # getting coordinates of region
        coords=region.coords
        size_coords=len(coords)
        # ---DEV--- adding partial HOG feature
        feature_vector[i] = _get_bins(orientation,magnitude,region)

        # ---DEV--- try on every channel, add different channels to features, ...
        # intensity_image uses bounding box, might be better to really compute gradient and magnitude 
        # for whole image and then compute bins directly using this information, without normalization?
        '''to_hog = region.intensity_image
        fd, hog_image = hog(to_hog, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, multichannel=False, feature_vector=True)
        fd=numpy.reshape(fd,(fd.shape[0]//9,9))
        # ---DEV--- is this meaningful to take the mean of a normalized array? 
        # ---DEV--- else extract gradient and magnitude and construct bins manually
        fd_bins=fd.mean(axis=0)
        # ---DEV--- find out why does this happen? (maybe one orientation is nan?)
        if(numpy.isnan(fd_bins[0])):
            feature_vector[i].extend([0,0,0,0,0,0,0,0,0])
        else:
            feature_vector[i].extend(fd_bins)'''

    return feature_vector

def _normalized_hog(labels,image):
    regions=measure.regionprops(labels,intensity_image=image)
    feature_vector=[ list() for _ in range(len(regions)) ]

    for i,region in enumerate(regions):
        # ---DEV--- try on every channel, add different channels to features, ...
        # intensity_image uses bounding box, might be better to really compute gradient and magnitude 
        # for whole image and then compute bins directly using this information, without normalization?
        to_hog = region.intensity_image
        fd, hog_image = hog(to_hog, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, multichannel=False, feature_vector=True)
        fd=numpy.reshape(fd,(fd.shape[0]//9,9))
        # ---DEV--- is this meaningful to take the mean of a normalized array? 
        # ---DEV--- else extract gradient and magnitude and construct bins manually
        fd_bins=fd.mean(axis=0)
        # ---DEV--- find out why does this happen? (maybe one orientation is nan?)
        if(numpy.isnan(fd_bins[0])):
            feature_vector[i].extend([0,0,0,0,0,0,0,0,0])
        else:
            feature_vector[i].extend(fd_bins)

    return feature_vector

# FIXME: is there something built-in in skimage?
def _color_features(labels,image_lab,image):
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
    mean_rgb, stdev_rgb=[0]*number_regions, [0]*number_regions
    feature_vector=[0]*number_regions

    for i,region in enumerate(regions):
        # getting coordinates of region
        coords=region.coords
        size_coords=len(coords)
        L_value, a_value, b_value=[0]*len(coords),[0]*len(coords),[0]*len(coords)
        #R_value, G_value, B_value=[0]*len(coords),[0]*len(coords),[0]*len(coords)
        for (l,(x,y)) in enumerate(coords):
            L,a,b=image_lab[(x,y)]
            L_value[l]=L
            a_value[l]=a
            b_value[l]=b
            # ---DEV--- doing the same for RGB color space
            '''R,G,B=image[(x,y)]
            R_value[l]=R
            G_value[l]=G
            B_value[l]=B'''
        # FIXME: statistics functions are very slow: try with numpy?
        mean_lab[i]=[sum(L_value)/size_coords,sum(a_value)/size_coords,sum(b_value)/size_coords]
        stdev_lab[i]=[numpy.std(L_value),numpy.std(a_value),numpy.std(b_value)]
        feature_vector[i]=mean_lab[i]+stdev_lab[i]
        # ---DEV--- RGB killed results: MAYBE THIS IS BECAUSE THE SCALES ARE VERY DIFFERENT?
        # ---DEV--- or mayve this is because HOG is not computed on corresponding channels?! ...NO SINCE GRAYSCALE
        '''mean_rgb[i]=[sum(R_value)/size_coords,sum(G_value)/size_coords,sum(B_value)/size_coords]
        stdev_rgb[i]=[numpy.std(R_value),numpy.std(G_value),numpy.std(B_value)]
        feature_vector[i]=mean_rgb[i]+stdev_rgb[i]'''

    return feature_vector

# FIXME: is there something built-in in skimage?
def _update_color_features(coords,image_lab,image):
    """
    Function that computes a feature vector for a given image and a set of segments

    :param labels:
        The segmentation to start from
    :param image_lab:
        The image in L*a*b* space
    """
    mean_lab, stdev_lab=list(), list()

    # getting coordinates of region
    size_coords=len(coords)
    L_value, a_value, b_value=[0]*len(coords),[0]*len(coords),[0]*len(coords)
    #R_value, G_value, B_value=[0]*len(coords),[0]*len(coords),[0]*len(coords)
    for (l,(x,y)) in enumerate(coords):
        L,a,b=image_lab[(x,y)]
        L_value[l]=L
        a_value[l]=a
        b_value[l]=b
        # ---DEV--- doing the same for RGB color space
        '''R,G,B=image[(x,y)]
        R_value[l]=R
        G_value[l]=G
        B_value[l]=B'''
    # FIXME: statistics functions are very slow: try with numpy?
    mean_lab=[sum(L_value)/size_coords,sum(a_value)/size_coords,sum(b_value)/size_coords]
    stdev_lab=[numpy.std(L_value),numpy.std(a_value),numpy.std(b_value)]
    # ---DEV--- RGB killed results: MAYBE THIS IS BECAUSE THE SCALES ARE VERY DIFFERENT?
    # ---DEV--- or mayve this is because HOG is not computed on corresponding channels?! ...NO SINCE GRAYSCALE
    '''mean_rgb[i]=[sum(R_value)/size_coords,sum(G_value)/size_coords,sum(B_value)/size_coords]
    stdev_rgb[i]=[numpy.std(R_value),numpy.std(G_value),numpy.std(B_value)]
    feature_vector[i]=mean_rgb[i]+stdev_rgb[i]'''

    return mean_lab+stdev_lab

# FIXME: add Davies-Bouldin and Caliński-Harabasz
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

        # ===== START DEV =====
        regions = measure.regionprops(g._segmentation_merged)
        for region in regions:
            xy = region.centroid
            x = xy[1]
            y = xy[0]
            text = ax[2][0].text(x, y, region.label,ha="center", va="center", color="w")
        # ===== END DEV ===== 

        for a in ax.ravel():
            a.set_axis_off()
    
    plt.tight_layout()
    plt.show()

# ===== COMPARISON FUNCTIONS ===== 
def _get_groundtruth(filepath):
    from scipy.io import loadmat
    groundtruth = loadmat(filepath)
    boundaries = []
    segmentation = []
    for i in range(len(groundtruth['groundTruth'][0])):
        # groundtruths boundaries and segmentation as numpy arrays
        boundaries.append(groundtruth['groundTruth'][0][i][0]['Boundaries'][0])
        segmentation.append(groundtruth['groundTruth'][0][i][0]['Segmentation'][0])
    return boundaries, segmentation

