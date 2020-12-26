# FIXME: use local imports instead
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

from numpy import asarray, unique, copy
from scipy.spatial.distance import cosine
from networkx import contracted_nodes

class GeST:
    # TODO: describe required arguments and optional ones
    def __init__(self, *args, **kwargs):
        # path to image
        self._path_to_image = args[0]
        self._n_cluster = args[1]

        # L*a*b* image
        self._image = io.imread(self._path_to_image)
        self._image = img_as_float(self._image)
        self._image_lab = (color.rgb2lab(self._image) + [0,128,128]) #// [1,1,1]

        self._preseg_method = kwargs.get("preseg_method", None)
        self._presegmentation = kwargs.get("presegmentation", None)
        self._embeddings = kwargs.get("embeddings", None)

        # TODO: allow for external classes to deal with initial preseg
        self._hs = 7
        self._hr = 4.5
        self._M = 50
        self._sigma = 125
        self._number_of_regions = 0

        # in case no presegmentation labels are provided 
        # FIXME: offer different possibilities according to method
        if(self._presegmentation is None):
            self.compute_preseg()

        self._RAG = None
        if(self._embeddings is None):
            self.compute_embeddings()

        # do we need to instantiate every class attribute?
        self._segmentation = None
        self._segmentation_merged = False
        self._clustering = None
        self._FV = None

    def set_msp_parameters(self,_hs,_hr,_M):
        self._hs = _hs
        self._hr = _hr
        self._M = _M

    # FIXME: different computation according to method used
    def compute_preseg(self):
        import time, sys
        begin = time.process_time()
        ms_image = imread(self._path_to_image)
        # TODO: try using Quickshift (from skimage) instead
        (_, labels, self._number_of_regions) = segment(ms_image, spatial_radius=self._hs, range_radius=self._hr, min_density=self._M)
        self._presegmentation = 1+labels
        end = time.process_time()
        print("presegmentation computed in {} seconds".format(end-begin), file=sys.stderr)

    def compute_embeddings(self):
        import time, sys
        # computing RAG
        
        begin = time.process_time() 
        self._RAG = graph.rag_mean_color(self._image_lab,self._presegmentation,connectivity=2,mode='similarity',sigma=self._sigma)
        end = time.process_time()
        print("RAG computed in {} seconds".format(end-begin), file=sys.stderr)

        # computing embeddings
        begin = time.process_time() 
        Gn2v = nv.Graph(self._RAG, False, 2, .5)
        Gn2v.preprocess_transition_probs()
        walks = Gn2v.simulate_walks(20, 20)
        # learn embeddings by optimizing the Skipgram objective using SGD.
        walks = [list(map(str, walk)) for walk in walks]
        # FIXME: allow parameterization
        model = Word2Vec(walks, size=32, window=5, min_count=0, sg=1, workers=4, iter=1)

        # getting the embeddings
        begin = time.process_time() 
        representation = model.wv
        nodes=self._RAG.nodes()
        self._embeddings = [representation.get_vector(str(node)).tolist() for node in nodes]
        end = time.process_time()
        print("embeddings computed in {} seconds".format(end-begin), file=sys.stderr)

    # TODO: "this is Algorithm 1 from the paper" + cut into functions
    def segmentation(self):
        import time, sys

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
        # building flat segmentation and then reshaping
        #self._segmentation=[ [ self._clustering[value-1]+1 for value in line ] for line in self._presegmentation ] 
        self._segmentation=asarray([self._clustering[value-1]+1 for line in self._presegmentation for value in line]).reshape(self._presegmentation.shape)
        end = time.process_time()
        print("clustering computed in {} seconds".format(end-begin), file=sys.stderr)

    def merge(self,thr_pixels=200,thr=0.995,sigma=5):
        import time, sys
        # NOTE; labels must be a matrix-like imaeg
        begin = time.process_time()
        labels_merged = copy(self._segmentation)
        merged=True
        has_merged=False
        # initial computation, will be maintained during algorithm
        G = graph.RAG(labels_merged,connectivity=1)
        while(merged):
            regions = measure.regionprops(self._segmentation)
            # FIXME: totally useless to compute again the ones that have not changed
            merged=False
            
            def _findregion(R):
                for i in range(len(regions)):
                    if regions[i].label == R:
                        return i
            
            for u,v in G.edges():
                Ri=regions[_findregion(u)]
                Rj=regions[_findregion(v)]
                sim=1-cosine(self._FV[Ri.label-1],self._FV[Rj.label-1])
                if sim >= thr:
                    #print("similarity merging region {} and {}.".format(Ri.label,Rj.label))
                    max_label = Ri if Ri.label > Rj.label else Rj
                    min_label = Ri if Ri.label < Rj.label else Rj
                    for (x,y) in max_label.coords:
                        labels_merged[(x,y)] = min_label.label
                    merged=True
                    has_merged=True
                    self._FV[min_label.label-1] = (self._FV[min_label.label-1]+self._FV[max_label.label-1])/2
                    G = contracted_nodes(G,min_label.label,max_label.label,self_loops=False)
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
                    closest = max([(regions[_findregion(Rj)].label,1-cosine(self._FV[Ri.label-1],self._FV[regions[_findregion(Rj)].label-1])) for Rj in neighbors],key=lambda x: x[1])[0]
                    Rj = regions[_findregion(closest)]
                    max_label = Ri if Ri.label > Rj.label else Rj
                    min_label = Ri if Ri.label < Rj.label else Rj
                    # could this actually be enough?
                    #max_label.label = min_label.label
                    for (x,y) in max_label.coords:
                        labels_merged[(x,y)] = min_label.label
                    merged=True
                    has_merged=True
                    # updating feature vector
                    self._FV[min_label.label-1] = (self._FV[min_label.label-1]+self._FV[max_label.label-1])/2
                    G = contracted_nodes(G,min_label.label,max_label.label,self_loops=False)
                if(merged):
                    break
            if(merged):
                continue
            
        _, labels_merged = unique(labels_merged,return_inverse=1)
        labels_merged=(1+labels_merged).reshape(self._presegmentation.shape)
        end = time.process_time()
        print("merging procedure done in {} seconds".format(end-begin))
        self._segmentation = labels_merged

