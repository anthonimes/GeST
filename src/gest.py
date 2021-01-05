from skimage import io, color, measure
from skimage.future import graph
from skimage.util import img_as_float

from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from gensim.models import Word2Vec

# https://github.com/fjean/pymeanshift
from pymeanshift import segment
from cv2 import imread
from src.utils.node2vec.src import node2vec as nv
from src.helper import _color_features, silhouette 

from numpy import asarray, unique, amax, copy, argwhere, zeros
from scipy.spatial.distance import cosine
from networkx import contracted_nodes, connected_components, is_connected

class GeST:
    """
    GeST object. Example:
        
        g = GeST('examples/86068.jpg', 24)
        g.segmentation()

    :param path_to_image:
        The (relative or absolute) path to the image to segment.
    :type first: ``str``
    :param n_cluster:
        The number of segments needed. If None, will be computed automatically. 
    :type second: ``int``
    :param \**kwargs:
        See below

    :Keywords arguments:
        * *preseg_method* (``str``) --
            Presegmentation method. Currently supported : MeanShift (``MS``).
        * *presegmentation* (``matrix``) -- 
            Initial presegmentation: matrix-like structure where each pixel is assigned to a given segment. 
            Labels of segments must range from 1 to the number of segments.
        * *embeddings* (``matrix``) -- 
            Initial embeddings computed from the RAG associated to the presegmentation. 
            The matrix must be nxd where ``n`` is the number of segments of the presegmentation, and ``d`` the dimension of the embeddings.
    """
        
    def __init__(self, path_to_image=None, n_cluster=None, **kwargs):
        self._path_to_image = path_to_image
        self._n_cluster = n_cluster

        # L*a*b* image
        self._image = io.imread(self._path_to_image)
        self._image = img_as_float(self._image)
        self._image_lab = (color.rgb2lab(self._image) + [0,128,128]) #// [1,1,1]

        self._preseg_method = kwargs.get("preseg_method", "MS")
        self._presegmentation = kwargs.get("presegmentation", None)
        self._embeddings = kwargs.get("embeddings", None)

        self._docontiguous = kwargs.get("contiguous", False)
        self._domerge = kwargs.get("merge", False)

        self._hs = kwargs.get("spatial_radius", 7)
        self._hr = kwargs.get("spatial_range",4.5)
        self._M = kwargs.get("min_density",50)
        self._sigma = kwargs.get("sigma",125)
        self._number_of_regions = 0

        self._RAG = None
        self._merged_RAG = None
        self._segmentation = None
        self._segmentation_merged = None
        self._clustering = None
        self._FV = None

        # if no presegmentation labels are provided 
        # TODO: offer different possibilities according to method
        if(self._presegmentation is None):
            self.compute_preseg()

        # if no embeddings are provided
        if(self._embeddings is None):
            self.compute_embeddings()

    # FIXME: different computation according to method used
    def compute_preseg(self):
        """
        Compute the initial presegmentation using ``preseg_method``
        """
        import time, sys
        begin = time.process_time()
        ms_image = imread(self._path_to_image)
        # TODO: try using Quickshift (from skimage) instead
        (_, labels, self._number_of_regions) = segment(ms_image, spatial_radius=self._hs, range_radius=self._hr, min_density=self._M)
        self._presegmentation = 1+labels
        end = time.process_time()
        print("presegmentation computed in {} seconds".format(end-begin), file=sys.stderr)

    def compute_embeddings(self):
        """
        Compute the RAG and embeddings from the initial presegmentation
        """
        import time, sys
        
        # computing RAG
        # TODO: add exception/error if _presegmentation is None
        begin = time.process_time() 
        self._RAG = graph.rag_mean_color(self._image_lab,self._presegmentation,connectivity=2,mode='similarity',sigma=self._sigma)
        end = time.process_time()
        print("RAG computed in {} seconds".format(end-begin), file=sys.stderr)

        # computing embeddings
        begin = time.process_time() 
        Gn2v = nv.Graph(self._RAG, False, 2, .5)
        Gn2v.preprocess_transition_probs()
        walks = Gn2v.simulate_walks(20, 20)
        walks = [list(map(str, walk)) for walk in walks]
        # FIXME: allow parameterization
        model = Word2Vec(walks, size=16, window=5, min_count=0, sg=1, workers=4, iter=1)

        begin = time.process_time() 
        representation = model.wv
        nodes=self._RAG.nodes()
        self._embeddings = [representation.get_vector(str(node)).tolist() for node in nodes]
        end = time.process_time()
        print("embeddings computed in {} seconds".format(end-begin), file=sys.stderr)

    def _contiguous(self):
        """
        (Private) Procedure that produce a contiguous set of segments. By default clustering on embeddings may provide 
        segments that are far apart within the image. 
        """
        Gr = graph.RAG(self._presegmentation, connectivity=1)

        new_labels = copy(self._clustering)
        for _label in unique(self._clustering):
            labelmax = amax(new_labels)
            # getting regions with this label
            vertices = 1+argwhere(new_labels == _label).flatten()
            Gc = Gr.subgraph(vertices)
            if(not(is_connected(Gc))):
                connected_component = sorted(connected_components(Gc), key=len, reverse=True)
                to_relabel=connected_component[1:]
                labelcpt=1
                for cc in to_relabel:
                    for vertex in cc:
                        new_labels[vertex-1]=labelmax+labelcpt
                    labelcpt+=1

        self._clustering = new_labels
        for l,line in enumerate(self._presegmentation):
            for j,value in enumerate(line):
                self._segmentation[l][j] = new_labels[value-1]+1

    # small regions merging --- noise removal
    def _pixels_merge(self,regions,thr_pixels=750):
        """
        (Private) Procedure that merge small segments with their closest neighbor.

        :param regions:
            The properties of the initially computed regions.
        :param thr_pixels:
            The threshold size for merging.
        """

        def _findregion(R):
            for i in range(len(regions)):
                if regions[i].label == R:
                    return i

        # trying to merge small regions to their most similar neighbors
        # FIXME: IS IT BETTER AFTER OR BEFORE MERGING SMALL REGIONS?
        for i in range(len(regions)):
            Ri = regions[i]
            lenRi = len(Ri.coords)
            if(lenRi < thr_pixels):
                # WARNING: neighbors in graphs are labels, not indices of regions array!
                neighbors = list(self._merged_RAG.neighbors(Ri.label))
                closest = max([(regions[_findregion(Rj)].label,self._merged_RAG[Ri.label][regions[_findregion(Rj)].label]['weight']) for Rj in neighbors], key=lambda x: x[1])[0]
                #closest = max([(regions[_findregion(Rj)].label,1-cosine(self._FV[Ri.label-1],self._FV[regions[_findregion(Rj)].label-1])) for Rj in neighbors],key=lambda x: x[1])[0]
                Rj = regions[_findregion(closest)]
                max_label = Ri if Ri.label > Rj.label else Rj
                min_label = Ri if Ri.label < Rj.label else Rj
                # could this actually be enough?
                #max_label.label = min_label.label
                for (x,y) in max_label.coords:
                    self._segmentation_merged[(x,y)] = min_label.label
                # updating feature vector
                #self._FV[min_label.label-1] = (self._FV[min_label.label-1]+self._FV[max_label.label-1])/2
                self._merged_RAG = contracted_nodes(self._merged_RAG,min_label.label,max_label.label,self_loops=False)
                return True
        return False

    def _similarity_merge(self,regions,thr=0.65):
        """
        (Private) Procedure that merge similar segments 

        :param regions:
            The properties of the initially computed regions.
        :param thr:
            The threshold for merging. This value depends on the distance considered. 
        """

        def _findregion(R):
            for i in range(len(regions)):
                if regions[i].label == R:
                    return i

        for u,v in self._merged_RAG.edges():
            Ri=regions[_findregion(u)]
            Rj=regions[_findregion(v)]
            #sim=1-cosine(self._FV[Ri.label-1],self._FV[Rj.label-1])
            sim = self._merged_RAG[u][v]['weight']
            if sim >= thr:
                #print("similarity merging region {} and {}.".format(Ri.label,Rj.label))
                max_label = Ri if Ri.label > Rj.label else Rj
                min_label = Ri if Ri.label < Rj.label else Rj
                for (x,y) in max_label.coords:
                    self._segmentation_merged[(x,y)] = min_label.label
                #self._FV[min_label.label-1] = (self._FV[min_label.label-1]+self._FV[max_label.label-1])/2
                self._merged_RAG = contracted_nodes(self._merged_RAG,min_label.label,max_label.label,self_loops=False)
                return True
        return False
        
        '''regions_merged = [False]*(len(regions)+1)
        for u,v in G.edges():
            Ri=self._findregion(u,regions)
            Rj=self._findregion(v,regions)
            if not(regions_merged[Ri.label] or regions_merged[Rj.label]):
                #sim=1-cosine(self._FV[Ri.label-1],self._FV[Rj.label-1])
                sim=G[u][v]['weight']
                if sim >= thr:
                    #print("similarity merging region {} and {}.".format(Ri.label,Rj.label))
                    R_max_label = Ri if Ri.label > Rj.label else Rj
                    R_min_label = Ri if Ri.label < Rj.label else Rj
                    regions_merged[R_max_label.label]=True
                    regions_merged[R_min_label.label]=True
                    for (x,y) in R_max_label.coords:
                        self._segmentation_merged[(x,y)] = R_min_label.label
                    # updating feature vector
                    #self._FV[R_min_label.label-1] = (self._FV[R_min_label.label-1]+self._FV[R_max_label.label-1])/2
                    # merging nodes in the RAG
                    G = contracted_nodes(G,R_min_label.label,R_max_label.label,self_loops=False)'''

    def _merge(self,thr_pixels=750,thr=0.65):
        """
        (Private) Procedure that merge while possible. First pixels, then similarity. 
        This is Algorithm 2 from GeSt: a new image segmentation technique based on graph embedding.
        
        :param thr_pixels:
            The threshold size for merging.
        :param thr:
            The threshold for merging. This value depends on the distance considered. 
        """

        import time, sys
        begin = time.process_time()
        if(self._segmentation_merged is None):
            self._segmentation_merged = copy(self._segmentation)
        # initial computation, will be maintained during algorithm
        self._merged_RAG = graph.rag_mean_color(self._image_lab,self._segmentation_merged,connectivity=2,mode='similarity',sigma=self._sigma)
        #G = graph.RAG(self._segmentation_merged,connectivity=1)

        while(True):
            regions = measure.regionprops(self._segmentation_merged)
            merged = self._similarity_merge(regions,thr)
            if(merged):
                continue
            merged = self._pixels_merge(regions,thr_pixels)
            if(merged):
                continue
            break

        _, self._segmentation_merged = unique(self._segmentation_merged,return_inverse=1)
        self._segmentation_merged=(1+self._segmentation_merged).reshape(self._presegmentation.shape)
        end = time.process_time()
        print("merging procedure done in {} seconds".format(end-begin))

    def segmentation(self):
        """
        Member method that implements Algorithm 1 of the paper GeSt: a new image segmentation technique based on graph embedding.
        """

        import time, sys
        import matplotlib.pyplot as plt

        # NOTE: Mean is included in graph somehow?
        begin = time.process_time() 
        self._FV = normalize(_color_features(self._presegmentation,self._image_lab))
        for l,v in enumerate(self._FV):
            self._embeddings[l].extend(v)
        end = time.process_time()
        print("feature vector computed in {} seconds".format(end-begin), file=sys.stderr)

        # clustering
        begin = time.process_time() 
        scaler = StandardScaler()
        data = scaler.fit_transform(self._embeddings)
            
        if(self._n_cluster is None):
            self._n_cluster = min(silhouette(data,25),self._number_of_regions)
        
        # using agglomerative clustering to obtain segmentation 
        clustering = AgglomerativeClustering(n_clusters=self._n_cluster,affinity='cosine',linkage='average',distance_threshold=None).fit(data)
        self._clustering = clustering.labels_
        end = time.process_time()
        print("clustering computed in {} seconds".format(end-begin), file=sys.stderr)

        # building flat segmentation and then reshaping
        self._segmentation=asarray([self._clustering[value-1]+1 for line in self._presegmentation for value in line]).reshape(self._presegmentation.shape)
        self._number_of_regions = len(unique(self._segmentation))

        if(self._docontiguous):
            self._contiguous()
            self._number_of_regions = len(unique(self._segmentation))
        if(self._domerge):
            # FIXME: should be parameters of __init()__
            self._merge(thr_pixels=750,thr=0.65)
            self._number_of_regions = len(unique(self._segmentation_merged))
        print("final segmentation has {} regions".format(self._number_of_regions))

    def dev_merge(self,thr_pixels=250,thr=0.998,sigma=125):
        import time, sys
        begin = time.process_time()
        self.merge_pixels(thr_pixels)
        self.merge_cosine(thr)
        end = time.process_time()
        print("merging procedure done in {} seconds".format(end-begin))

