from skimage import io,color,measure,img_as_ubyte
from skimage.segmentation import mark_boundaries

from math import sqrt, ceil
import random

import sklearn.metrics
import sklearn.cluster

import numpy
import argparse

def _parse_args():    
    # TODO: add argument for contiguity
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
    regions=measure.regionprops(segmentation)
    # computing masks to apply to region histograms
    # loop over the unique segment values
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

def _savepreseg(segmentation=None,image=None,path=None,name=None):
    io.imsave(path,img_as_ubyte(mark_boundaries(image,segmentation, mode='thick')))

def _colors_by_region(N):
    return [(random.random(), random.random(), random.random()) for e in range(0,256,ceil(256//N))]

def _savefig(segmentation=None,image=None,path=None,name=None):
    colored_regions=color.label2rgb(segmentation, image, alpha=1, colors=_colors(segmentation,image), bg_label=0)
    io.imsave(path,img_as_ubyte(colored_regions))
    #colored_by_regions=color.label2rgb(segmentation, image, alpha=1, colors=_colors_by_region(numpy.amax(segmentation)), bg_label=0)
    #io.imsave(path[:-4]+"_COLORMAP"+path[-4:],img_as_ubyte(mark_boundaries(colored_by_regions, segmentation, mode='thick')))

def _loadlabels(filename):
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            labels.append(list(map(int,line.strip().split('\t'))))
    return numpy.asarray(labels)

def _loadembeddings(path):
    return numpy.load(path)

# FIXME: is there something built-in in skimage?
def _color_features(labels,image_lab):
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
        # FIXME: statistics functions are very slow: try with numpy?
        mean_lab[i]=[sum(L_value)/size_coords,sum(a_value)/size_coords,sum(b_value)/size_coords]

        def variance(data):
            n=len(data)
            mean=sum(data) / n
            deviations=[(x - mean) ** 2 for x in data]
            variance=sum(deviations) / n
            return variance

        #stdev_lab[i]=[sqrt(variance(L_value)),sqrt(variance(a_value)),sqrt(variance(b_value))]
        stdev_lab[i]=[numpy.std(L_value),numpy.std(a_value),numpy.std(b_value)]
        feature_vector[i]=mean_lab[i]+stdev_lab[i]

    return feature_vector

def silhouette(points,kmax):
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

