from skimage import data,io,color,filters,measure,util,img_as_ubyte
from skimage.segmentation import mark_boundaries
from skimage.future import graph

from scipy.io import loadmat, savemat
from scipy.special import comb
from scipy.spatial import distance
from sklearn.preprocessing import normalize

from statistics import mean, stdev
from math import exp, log, ceil

from os import walk

# https://github.com/fjean/pymeanshift
import pymeanshift as pms
import argparse, random
import numpy, cv2
import matplotlib.pyplot as plt
import networkx as nx

def _parse_args():    
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
    return vars(ap.parse_args())

def _get_groundtruth(filepath):
    groundtruth = loadmat(filepath)
    boundaries = []
    segmentation = []
    for i in range(len(groundtruth['groundTruth'][0])):
        # groundtruths boundaries and segmentation as numpy arrays
        boundaries.append(groundtruth['groundTruth'][0][i][0]['Boundaries'][0])
        segmentation.append(groundtruth['groundTruth'][0][i][0]['Segmentation'][0])
    return boundaries, segmentation

def _get_BSR(filepath):
    BSR = loadmat(filepath)
    return BSR['seg']

def _savemat(filepath,segmentation):
    savemat(filepath,{"segs": segmentation},appendmat=False)
    
def _colors_by_region(N):
    return [(random.random(), random.random(), random.random()) for e in range(0,256,ceil(256//N))]

def _colors(segmentation,image):
    regions = measure.regionprops(segmentation)
    # computing masks to apply to region histograms
    # loop over the unique segment values
    colors = [0]*len(regions)
    for index,region in enumerate(regions):
        # getting coordinates of region
        coords = region.coords
        #cy, cx = region.centroid
        #plt.plot(cx, cy, 'ro')
        #plt.show()
        R_value, G_value, B_value=[0]*len(coords),[0]*len(coords),[0]*len(coords)
        for p,(x,y) in enumerate(coords):
            R,G,B=image[(x,y)]
            R_value[p]=R
            G_value[p]=G
            B_value[p]=B
        colors[index]=(mean(R_value),mean(G_value),mean(B_value))
    return colors

def _savepreseg(segmentation=None,image=None,path=None,name=None):
    io.imsave(path,img_as_ubyte(mark_boundaries(image,segmentation, mode='thick')))
    #plt.axis("off")
    #plt.savefig(path,bbox_inches='tight',pad_inches=0)'''

def _savefig(segmentation=None,image=None,path=None,name=None):
    #fig = plt.figure(name)#figsize=(image.shape[1]/80,image.shape[0]/80),dpi=80)
    #if(colored):
    colored_regions = color.label2rgb(segmentation, image, alpha=1, colors=_colors(segmentation,image), bg_label=0)
    io.imsave(path,img_as_ubyte(mark_boundaries(colored_regions, segmentation, mode='thick')))
    colored_by_regions = color.label2rgb(segmentation, image, alpha=1, colors=_colors_by_region(numpy.amax(segmentation)), bg_label=0)
    io.imsave(path[:-4]+"_COLORMAP"+path[-4:],img_as_ubyte(mark_boundaries(colored_by_regions, segmentation, mode='thick')))
    #plt.axis("off")
    #plt.savefig(path,bbox_inches='tight',pad_inches=0)'''

def _loadlabels(filename):
    labels = []
    with open(filename, "r") as f:
        for line in f:
            labels.append(list(map(int,line.strip().split('\t'))))
    return numpy.asarray(labels)

def _loadembeddings(path):
    return numpy.load(path)

def _color_features(labels,image_lab):
    regions = measure.regionprops(labels)
    mean_lab, stdev_lab = [0]*len(regions), [0]*len(regions)
    feature_vector = [0]*len(regions)
    #feature_vector=[]
    for i,region in enumerate(regions):
        # getting coordinates of region
        coords = region.coords
        L_value, a_value, b_value=[0]*len(coords),[0]*len(coords),[0]*len(coords)
        for (l,(x,y)) in enumerate(coords):
            L,a,b=image_lab[(x,y)]
            L_value[l]=L
            a_value[l]=a
            b_value[l]=b
        mean_lab[i]=[mean(L_value),mean(a_value),mean(b_value)]
        stdev_lab[i]=[stdev(L_value),stdev(a_value),stdev(b_value)]
        feature_vector[i]=mean_lab[i]+stdev_lab[i]
        
    '''HOG=[0]*len(regions)
    for i in range(len(regions)):
        bbox=regions[i].bbox
        segment=image_lab[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        # FIXME; how the fuck can we obtain a same-dimension vector?
        segment_resized=resize(segment,(64,128))
        fd = hog(segment_resized, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=False, multichannel=True, feature_vector=False)
        # each bin represents the orientation: 0--20, 20--40, ..., 160--180
        (a,b,c,d,_)=fd.shape
        descriptor=[0]*9
        for x in range(a):
            for y in range(b):
                for w in range(c):
                    for z in range(d):
                        for l in range(9):
                            orientation=fd[x,y,w,z,l]*10
                            descriptor[l]+=(orientation)

        descriptor = [d/(a*b*c*d) for d in descriptor]
        feature_vector.append(mean_lab[i]+stdev_lab[i])
        feature_vector[i].extend(descriptor)'''

    return feature_vector

def _get_Lab_adjacency(labels,image_lab,sigma=50):
        regions = measure.regionprops(labels)
        mean_lab, stdev_lab = [0]*len(regions), [0]*len(regions)
        for index,region in enumerate(regions):
            # getting coordinates of region
            coords = region.coords
            L_value, a_value, b_value=[0]*len(coords),[0]*len(coords),[0]*len(coords)
            for (i,(x,y)) in enumerate(coords):
                L,a,b=image_lab[(x,y)]
                L_value[i]=L
                a_value[i]=a
                b_value[i]=b
            mean_lab[index]=[mean(L_value),mean(a_value),mean(b_value)]
            stdev_lab[index]=[stdev(L_value),stdev(a_value),stdev(b_value)]

        adjacency = numpy.zeros((len(numpy.unique(labels))+1, len(numpy.unique(labels))+1))
        pairs=0
        total_pairs=0
        feature_vector=[]
        for i in range(1,len(adjacency)):
            for j in range(i+1,len(adjacency)):
                Li=mean_lab[i-1]#+stdev_lab[i-1]
                Lj = mean_lab[j-1]#+stdev_lab[j-1]
                Si=stdev_lab[i-1]
                Sj=stdev_lab[j-1]
                dist=color.deltaE_cie76(Li,Lj)
                distS=color.deltaE_cie76(Si,Sj)
                #gk=exp(-((dist**2)/(2*(sigma**2))))
                #gkS=exp(-((distS**2)/(2*(sigma**2))))
                gk=exp(-((dist**2)/(sigma)))
                #gkS=exp(-((distS**2)/(sigma)))
                #Hi = HOG[i-1].flatten().tolist()
                #Hj = HOG[j-1].flatten().tolist()
                '''tog = cosine(Hi,Hj)
                # NEED TO FIND THE RIGHT PARAMETER
                try:
                    sim = 0.2*(sqrt(gk*tog))+0.8*gk
                except:
                    sim=0'''
                #sim=gk*gkS
                sim=gk
                adjacency[i][j] = sim
                adjacency[j][i] = sim
        #where_are_NaNs = numpy.isnan(adjacency)
        #adjacency[where_are_NaNs] = 0
            
        return adjacency

def _distance_zero_graph(G,image=None,labels=None,threshold=15,sigma=50):
    adjacency = _get_Lab_adjacency(labels,image,sigma)
    Gr = nx.Graph(G)
    # removing edges below threshold
    maxdelta=0
    pairs=0
    edges = set()
    #print("starting from graph with {} vertices and {} edges".format(len(G),len(G.edges())))
    for u,v in Gr.edges():
        sim=adjacency[u][v]
        if(sim>=threshold): 
            Gr[u][v]['weight']=sim
        # commented to ensure connectivity
    
    # normalizing
#    maxsim=max(edges, key=lambda x: x[2])[2]
#    edges=[(e[0],e[1],maxsim-e[2]) for e in edges]
    Gr.add_weighted_edges_from(edges)
    return Gr

def _distance_r_graph(G,R,image=None,labels=None,threshold=15,sigma=50):
    adjacency = _get_Lab_adjacency(labels,image,sigma)
    Gr = nx.Graph(G)
    # removing edges below threshold
    maxdelta=0
    pairs=0
    edges = set()
    #print("starting from graph with {} vertices and {} edges".format(len(G),len(G.edges())))
    for u,v in Gr.edges():
        sim=adjacency[u][v]
        if(sim>=threshold): 
            Gr[u][v]['weight']=sim
        # commented to ensure connectivity
        else:
            Gr.remove_edge(u,v)
    if(R>0):
        for u in G.nodes():
            # we get its distance-R induced subgraph
            Ir = nx.single_source_shortest_path_length(G ,source=u, cutoff=R)
            # we then add an edge between u and every vertex in Ir, with proper weight
            for v in Ir.keys():
                if(u != v):
                    sim=adjacency[u][v]
                    if(sim>=threshold):
                        # a simple try: ponderate by distance (if similar but far apart, well...)
                        edges.add((u,v,sim))
                    # removing useless edges --- UGLY NEED TO BE FIXED
                    # commented to ensure connectivity
                    elif Gr.has_edge(u,v):
                        Gr.remove_edge(u,v)

    # normalizing
#    maxsim=max(edges, key=lambda x: x[2])[2]
#    edges=[(e[0],e[1],maxsim-e[2]) for e in edges]
    Gr.add_weighted_edges_from(edges)
    return Gr

def _f_measure(labels_ground_truth, labels_prediction):
    # tp = true positive, tn: true negative, fp: false positive, fn: false negative
    # number of pairs in the same set in ground truth
    sum_tp_fp = comb(numpy.bincount(labels_ground_truth), 2).sum()
    # number of pairs in the same set in prediction
    sum_tp_fn = comb(numpy.bincount(labels_prediction), 2).sum()
    # concatenating the results
    A = numpy.c_[(labels_ground_truth, labels_prediction)]
    tp = sum(comb(numpy.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in set(labels_ground_truth))
    fp = sum_tp_fp - tp
    fn = sum_tp_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return tp/(tp+(fp+fn)/2)

# np.bincount: this function is used to number of passing numbers was found.
# comb: combination example (6 2) = 15, (10, 2) = 45
# np_c: concanatenation operation.
# tp: every time a pair of elements is grouped together by the two cluster
# tn: every time a pair of elements is not grouped together by the two cluster
"""
    This function is used for calculate rand index (RI) score
    @param labels_ground_truth: actual label values
    @param labels_prediction: predicted label values
"""
def _rand_index_score(labels_ground_truth, labels_prediction):
    # tp = true positive, tn: true negative, fp: false positive, fn: false negative
    # number of pairs in the same set in ground truth
    sum_tp_fp = comb(numpy.bincount(labels_ground_truth), 2).sum()
    # number of pairs in the same set in prediction
    sum_tp_fn = comb(numpy.bincount(labels_prediction), 2).sum()
    # concatenating the results
    A = numpy.c_[(labels_ground_truth, labels_prediction)]
    tp = sum(comb(numpy.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in set(labels_ground_truth))
    fp = sum_tp_fp - tp
    fn = sum_tp_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)
"""
    This function is used to applyied probabilistic rand index evaluation metric.
    @param image_name: image name
    @param prediction: slic and region merge algoritms result
    @param score / number_of_ground_truth: PRI result for related image
"""
def _probabilistic_rand_index(groundtruth, prediction):
    score = 0
    number_of_ground_truth = len(groundtruth)
    for i in range(number_of_ground_truth):
        segmentation = groundtruth[i].flatten().tolist()
        score += _rand_index_score(segmentation, prediction.flatten().tolist())
    return score / number_of_ground_truth            

def _meanshift_py(path,_sr,_rr,_mind):
    ms_image = cv2.imread(path)
    (segmented_image, labels, number_regions) = pms.segment(ms_image, spatial_radius=_sr, range_radius=_rr, min_density=_mind)
    return 1+labels

def _merge_pixels(labels,image_lab,thr_pixels=300,sigma=5):
    # NOTE; labels must be a matrix-like image
    labels_merge = numpy.copy(labels)
    merged=True
    has_merged=False
    # initial computation, will be maintained during algorithm
    feature_vector = normalize(numpy.asarray(_color_features(labels,image_lab)))
    tomerge = numpy.unique(labels)
        
    for t in tomerge:
        Ri = t
        # how many pixels have this label?
        lenRi = len(numpy.where(labels_merge.flatten()==Ri)[0])
        if(lenRi > 0 and lenRi < thr_pixels):
            closest = max([(Rj,1-distance.cosine(feature_vector[Ri-1],feature_vector[Rj-1])) for Rj in tomerge if Ri!=Rj],key=lambda x: x[1])[0]
            Rj = closest
            sim=1-distance.cosine(feature_vector[Ri-1],feature_vector[Rj-1])
            #if(sim>=0.996):
            max_label = Ri if Ri > Rj else Rj
            min_label = Ri if Ri < Rj else Rj
            # updating remaining labels
            labels_merge[numpy.where(labels_merge==max_label)] = min_label
            #clusters_merge[numpy.where(clusters_merge==max_label-1)] = min_label-1
            has_merged=True
            feature_vector[min_label-1] = (feature_vector[min_label-1]+feature_vector[max_label-1])/2

    if(has_merged):
        _, labels_merge = numpy.unique(labels_merge,return_inverse=1)
        labels_merge=(1+labels_merge).reshape(labels.shape)
    return labels_merge,has_merged

# FIXME: try again to merge ALL images with small pixels, just to see improvements
def _merge_cosine(labels,image_lab,thr=0.999,sigma=5):
    # NOTE; labels must be a matrix-like image
    labels_merge = numpy.copy(labels)
    merged=True
    has_merged=False
    # initial computation, will be maintained during algorithm
    feature_vector = numpy.asarray(_color_features(labels,image_lab))
    G = graph.rag_mean_color(image_lab,labels,connectivity=2,mode='similarity',sigma=sigma)
        
    for u,v in G.edges():
        Ri = u
        Rj = v
        # is the closest region close enough?
        sim=1-distance.cosine(feature_vector[Ri-1],feature_vector[Rj-1])
        #sim=G[Ri][Rj]['weight']
        if(sim>=thr):
            max_label = Ri if Ri > Rj else Rj
            min_label = Ri if Ri < Rj else Rj
            # updating remaining labels
            labels_merge[numpy.where(labels_merge==max_label)] = min_label
            #clusters_merge[numpy.where(clusters_merge==max_label-1)] = min_label-1
            has_merged=True
            feature_vector[min_label-1] = (feature_vector[min_label-1]+feature_vector[max_label-1])/2
            #G = nx.contracted_nodes(G,min_label,max_label,self_loops=False)
    
    if(has_merged):
        _, labels_merge = numpy.unique(labels_merge,return_inverse=1)
        labels_merge=(1+labels_merge).reshape(labels.shape)
    return labels_merge,has_merged

def _merge(labels,image_lab,thr_pixels=200,thr=0.995,sigma=5):
    # NOTE; labels must be a matrix-like imaeg
    labels_merge = numpy.copy(labels)
    merged=True
    has_merged=False
    # initial computation, will be maintained during algorithm
    feature_vector = normalize(numpy.asarray(_color_features(labels_merge,image_lab)))
    G = graph.RAG(labels_merge,connectivity=1)
    while(merged):
        regions = measure.regionprops(labels_merge)
        # FIXME: totally useless to compute again the ones that have not changed...
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
                # updating remaining labels
                #_, labels_merge = numpy.unique(labels_merge,return_inverse=1)
                #labels_merge=(1+labels_merge).reshape(labels.shape)
                # updating feature vector
                feature_vector[min_label.label-1] = (feature_vector[min_label.label-1]+feature_vector[max_label.label-1])/2
                G = nx.contracted_nodes(G,min_label.label,max_label.label,self_loops=False)
                #print("COSI",(feature_vector[min_label.label-1]+feature_vector[max_label.label-1])/2)
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
                # updating remaining labels
                #_, labels_merge = numpy.unique(labels_merge,return_inverse=1)
                #labels_merge=(1+labels_merge).reshape(labels.shape)
                # updating feature vector
                #print("PIXE",(feature_vector[min_label.label-1]+feature_vector[max_label.label-1])/2)
                feature_vector[min_label.label-1] = (feature_vector[min_label.label-1]+feature_vector[max_label.label-1])/2
                G = nx.contracted_nodes(G,min_label.label,max_label.label,self_loops=False)
            if(merged):
                break
        if(merged):
            continue
        
    _, labels_merge = numpy.unique(labels_merge,return_inverse=1)
    labels_merge=(1+labels_merge).reshape(labels.shape)
    return labels_merge,has_merged
