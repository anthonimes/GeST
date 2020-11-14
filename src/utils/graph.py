import networkx as nx
from skimage import measure,color,data
from math import sqrt,exp
from statistics import mean, stdev
import numpy,cv2
from scipy.spatial.distance import euclidean, cosine, cityblock
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity

from skimage.transform import rescale, resize, downscale_local_mean

from skimage.feature import hog

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
                gkS=exp(-((distS**2)/(sigma)))
                #Hi = HOG[i-1].flatten().tolist()
                #Hj = HOG[j-1].flatten().tolist()
                '''tog = cosine(Hi,Hj)
                # NEED TO FIND THE RIGHT PARAMETER
                try:
                    sim = 0.2*(sqrt(gk*tog))+0.8*gk
                except:
                    sim=0'''
                sim=gk*gkS
                #sim=gk
                adjacency[i][j] = sim
                adjacency[j][i] = sim
        #where_are_NaNs = numpy.isnan(adjacency)
        #adjacency[where_are_NaNs] = 0
        for i in range(len(regions)):
            feature_vector.append(mean_lab[i]+stdev_lab[i])
            
        return adjacency, feature_vector

def _distance_zero_graph(G,image=None,adjacency=None,threshold=15):
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

def _distance_r_graph(G,R,image=None,adjacency=None,threshold=15):
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

if __name__=="__main__":
    image = io.imread(dirpath+filename)
    image = img_as_float(image)

    #image_lab = image
    image_lab = color.rgb2lab(image)
    image_lab = (color.rgb2lab(image) + [0,128,128]) #// [1,1,1]
