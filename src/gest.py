# FIXME: use local imports instead
from skimage import io, color
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

from numpy import asarray

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

    def set_msp_parameters(self,_hs,_hr,_M):
        self._hs = _hs
        self._hr = _hr
        self._M = _M

    # FIXME: different computation according to method used
    def compute_preseg(self):
        ms_image = imread(self._path_to_image)
        # TODO: try using Quickshift (from skimage) instead
        (_, labels, self._number_of_regions) = segment(ms_image, spatial_radius=self._hs, range_radius=self._hr, min_density=self._M)
        self._presegmentation = 1+labels

    def compute_embeddings(self):
        import time, sys
        # computing RAG
        
        self._RAG = graph.rag_mean_color(self._image_lab,self._presegmentation,connectivity=2,mode='similarity',sigma=self._sigma)

        # computing embeddings
        Gn2v = nv.Graph(self._RAG, False, 2, .5)
        Gn2v.preprocess_transition_probs()
        walks = Gn2v.simulate_walks(20, 20)
        # learn embeddings by optimizing the Skipgram objective using SGD.
        walks = [list(map(str, walk)) for walk in walks]
        # FIXME: allow parameterization
        model = Word2Vec(walks, size=32, window=5, min_count=0, sg=1, workers=4, iter=1)

        # getting the embeddings
        representation = model.wv
        nodes=self._RAG.nodes()
        self._embeddings = [representation.get_vector(str(node)).tolist() for node in nodes]

    # TODO: "this is Algorithm 1 from the paper" + cut into functions
    def segmentation(self):
        # NOTE: Mean is included in graph somehow?
        feature_vector = normalize(_color_features(self._presegmentation,self._image_lab))
        for l,v in enumerate(feature_vector):
            self._embeddings[l].extend(v)

        # clustering
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

