from skimage import io,color,measure,img_as_ubyte
from skimage.segmentation import mark_boundaries

from math import sqrt

import sklearn.metrics
import sklearn.cluster

import numpy
import argparse

def _parse_args():    
    # TODO: add argument for merge
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-p", "--path", required=True, help = "path to folder containing images")
    ap.add_argument("-m", "--method", required = False, default="msp", help="pre-segmentation method")
    ap.add_argument( "--sigma", required = False, help="kernel parameter", default=125)

    ap.add_argument("-n", "--nclusters", required = False, default=24, help="number of clusters")
    ap.add_argument("--silhouette", required = False, default=False, action="store_true", help="use silhouette method instead of fixed number of clusters")

    ap.add_argument("--hs", required = False, help="spatial radius", default=7)
    ap.add_argument("--hr", required = False, help="range radius", default=4.5)
    ap.add_argument( "--mind", required = False, help="min density", default=50)

    ap.add_argument("--merge", required = False, default=False, action="store_true", help="apply merging procedure")
    ap.add_argument("-w", "--write", required = False, default=False, action="store_true", help="write all files to hard drive")
    argsy = vars(ap.parse_args())

    write = argsy['write']
    silh = argsy['silhouette']
    merge = argsy['merge']

    n_cluster =  None if silh else int(argsy['nclusters'])
    sigma=float(argsy['sigma'])

    # TODO: allow for a single image or for path
    path_images = argsy['path']
    
    return argsy['method'], write, silh, merge, n_cluster, path_images, sigma


'''def _savemat(filepath,segmentation):
    savemat(filepath,{"segs": segmentation},appendmat=False)
   
# FIXME: should be internal methods
def _colors_by_region(N):
    return [(random.random(), random.random(), random.random()) for e in range(0,256,ceil(256//N))]'''

def _colors(segmentation,image):
    regions = measure.regionprops(segmentation)
    # computing masks to apply to region histograms
    # loop over the unique segment values
    colors = [0]*len(regions)
    for index,region in enumerate(regions):
        # getting coordinates of region
        coords = region.coords
        size_coords = len(coords)
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

def _savefig(segmentation=None,image=None,path=None,name=None):
    colored_regions = color.label2rgb(segmentation, image, alpha=1, colors=_colors(segmentation,image), bg_label=0)
    io.imsave(path,img_as_ubyte(mark_boundaries(colored_regions, segmentation, mode='thick')))
    #colored_by_regions = color.label2rgb(segmentation, image, alpha=1, colors=_colors_by_region(numpy.amax(segmentation)), bg_label=0)
    #io.imsave(path[:-4]+"_COLORMAP"+path[-4:],img_as_ubyte(mark_boundaries(colored_by_regions, segmentation, mode='thick')))

def _loadlabels(filename):
    labels = []
    with open(filename, "r") as f:
        for line in f:
            labels.append(list(map(int,line.strip().split('\t'))))
    return numpy.asarray(labels)

def _loadembeddings(path):
    return numpy.load(path)

# FIXME: is there something built-in in skimage?
def _color_features(labels,image_lab):
    regions = measure.regionprops(labels)
    number_regions=len(regions)
    mean_lab, stdev_lab = [0]*number_regions, [0]*number_regions
    feature_vector = [0]*number_regions
    for i,region in enumerate(regions):
        # getting coordinates of region
        coords = region.coords
        size_coords = len(coords)
        L_value, a_value, b_value=[0]*len(coords),[0]*len(coords),[0]*len(coords)
        for (l,(x,y)) in enumerate(coords):
            L,a,b=image_lab[(x,y)]
            L_value[l]=L
            a_value[l]=a
            b_value[l]=b
        # FIXME: statistics functions are very slow: try with numpy?
        mean_lab[i]=[sum(L_value)/size_coords,sum(a_value)/size_coords,sum(b_value)/size_coords]
        def variance(data):
            n = len(data)
            mean = sum(data) / n
            deviations = [(x - mean) ** 2 for x in data]
            variance = sum(deviations) / n
            return variance

        #stdev_lab[i]=[sqrt(variance(L_value)),sqrt(variance(a_value)),sqrt(variance(b_value))]
        stdev_lab[i]=[numpy.std(L_value),numpy.std(a_value),numpy.std(b_value)]
        feature_vector[i]=mean_lab[i]+stdev_lab[i]

    return feature_vector

# TODO: make sure this coincides with final version of article
'''def _merge(labels,image_lab,thr_pixels=200,thr=0.995,sigma=5):
    # NOTE; labels must be a matrix-like imaeg
    labels_merge = numpy.copy(labels)
    merged=True
    has_merged=False
    # initial computation, will be maintained during algorithm
    feature_vector = normalize(numpy.asarray(_color_features(labels_merge,image_lab)))
    G = graph.RAG(labels_merge,connectivity=1)
    while(merged):
        regions = measure.regionprops(labels_merge)
        # FIXME: totally useless to compute again the ones that have not changed
        merged=False
        
        def _findregion(R):
            for i in range(len(regions)):
                if regions[i].label == R:
                    return i
        
        for u,v in G.edges():
            Ri=regions[_findregion(u)]
            Rj=regions[_findregion(v)]
            sim=1-distance.cosine(feature_vector[Ri.label-1],feature_vector[Rj.label-1])
            if sim >= thr:
                #print("similarity merging region {} and {}.".format(Ri.label,Rj.label))
                max_label = Ri if Ri.label > Rj.label else Rj
                min_label = Ri if Ri.label < Rj.label else Rj
                for (x,y) in max_label.coords:
                    labels_merge[(x,y)] = min_label.label
                merged=True
                has_merged=True
                feature_vector[min_label.label-1] = (feature_vector[min_label.label-1]+feature_vector[max_label.label-1])/2
                G = nx.contracted_nodes(G,min_label.label,max_label.label,self_loops=False)
            if(merged):
                break
        if(merged):
            continue
                
        # trying to merge small regions to their most similar neighbors
        # FIXME: IS IT BETTER AFTER OR BEFORE MERGING SMALL REGIONS?
        for i in range(len(regions)):
            Ri = regions[i]
            lenRi = len(Ri.coords)
            if(lenRi < thr_pixels):
                # WARNING: neighbors in graphs are labels, not indices of regions array!
                neighbors = list(G.neighbors(Ri.label))
                closest = max([(regions[_findregion(Rj)].label,1-distance.cosine(feature_vector[Ri.label-1],feature_vector[regions[_findregion(Rj)].label-1])) for Rj in neighbors],key=lambda x: x[1])[0]
                Rj = regions[_findregion(closest)]
                max_label = Ri if Ri.label > Rj.label else Rj
                min_label = Ri if Ri.label < Rj.label else Rj
                # could this actually be enough?
                #max_label.label = min_label.label
                for (x,y) in max_label.coords:
                    labels_merge[(x,y)] = min_label.label
                merged=True
                has_merged=True
                # updating feature vector
                feature_vector[min_label.label-1] = (feature_vector[min_label.label-1]+feature_vector[max_label.label-1])/2
                G = nx.contracted_nodes(G,min_label.label,max_label.label,self_loops=False)
            if(merged):
                break
        if(merged):
            continue
        
    _, labels_merge = numpy.unique(labels_merge,return_inverse=1)
    labels_merge=(1+labels_merge).reshape(labels.shape)
    return labels_merge,has_merged'''

def silhouette(points,kmax):
    def SSE():
        sse=[]
        for k in range(2, kmax):
            km = sklearn.cluster.AgglomerativeClustering(n_clusters=k,affinity='cosine',linkage='average',distance_threshold=None).fit(points)
            labels_clustering = km.labels_
            silhouette_avg=sklearn.metrics.silhouette_score(points, labels_clustering, metric = 'cosine')
            sse.append(silhouette_avg)
        return sse
        
    scores = SSE()
    best = scores.index(max(scores))+2
    return best

