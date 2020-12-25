import skimage.io, skimage.color, skimage.future, skimage
import sklearn, sklearn.preprocessing
import cv2
import gensim
import pymeanshift
import numpy
import utils.node2vec.src.node2vec as nv

'''from skimage import io, color
from skimage.future import graph
from skimage.util import img_as_float

from sklearn import cluster as cl

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from utils.node2vec.src import node2vec
from gensim.models import Word2Vec

# https://github.com/fjean/pymeanshift
from pymeanshift import segment
from cv2 import imread'''

from helper import _color_features, silhouette

class GeST:
    # TODO: describe required arguments and optional ones
    def __init__(self, *args, **kwargs):
        # path to image
        self._path_to_image = args[0]
        self._n_cluster = args[1]

        # L*a*b* image
        self._image = skimage.io.imread(self._path_to_image)
        self._image = skimage.img_as_float(self._image)
        self._image_lab = (skimage.color.rgb2lab(self._image) + [0,128,128]) #// [1,1,1]

        self._preseg_method = kwargs.get("preseg_method", None)
        self._presegmentation = kwargs.get("presegmentation", None)

        # TODO: allow for external classes to deal with initial preseg
        self._hs = 7
        self._hr = 4.5
        self._M = 50
        self._sigma = 125
        self._number_of_regions = 0

        # THE PREFERED WAY IS TO PROVIDE LABELS FOR A PRESEGMENTATION
        if(self._preseg_method is not None):
            import time, sys
            self.compute_preseg()
            begin = time.process_time()
            end = time.process_time()
            print("presegmentation computed in {} seconds".format(end-begin), file=sys.stderr)

        # do we need to instantiate every class attribute?
        self._embeddings = None
        self._RAG = None
        self._segmentation = None
        self._segmentation_merged = None
        self._clustering = None

    def set_msp_parameters(self,_hs,_hr,_M):
        self._hs = _hs
        self._hr = _hr
        self._M = _M

    # FIXME: different computation according to method used
    def compute_preseg(self):
        ms_image = cv2.imread(self._path_to_image)
        # TODO: try using Quickshift (from skimage) instead
        (_, labels, self._number_of_regions) = pymeanshift.segment(ms_image, spatial_radius=self._hs, range_radius=self._hr, min_density=self._M)
        self._presegmentation = 1+labels

    # TODO: "this is Algorithm 1 from the paper" + cut into functions
    def segmentation(self):
        import time, sys
        # computing RAG
        
        begin = time.process_time() 
        self._RAG = skimage.future.graph.rag_mean_color(self._image_lab,self._presegmentation,connectivity=2,mode='similarity',sigma=self._sigma)
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
        model = gensim.models.Word2Vec(walks, size=32, window=5, min_count=0, sg=1, workers=4, iter=1)

        # getting the embeddings
        begin = time.process_time() 
        representation = model.wv
        nodes=self._RAG.nodes()
        self._embeddings = [representation.get_vector(str(node)).tolist() for node in nodes]
        end = time.process_time()
        print("embeddings computed in {} seconds".format(end-begin), file=sys.stderr)

        # NOTE: Mean is included in graph somehow?
        begin = time.process_time() 
        feature_vector = sklearn.preprocessing.normalize(_color_features(self._presegmentation,self._image_lab))
        for l,v in enumerate(feature_vector):
            self._embeddings[l].extend(v)
        end = time.process_time()
        print("feature vector computed in {} seconds".format(end-begin), file=sys.stderr)

        # clustering
        begin = time.process_time() 
        scaler = sklearn.preprocessing.StandardScaler()
        data = scaler.fit_transform(self._embeddings)
            
        if(self._n_cluster is None):
            self._n_cluster = min(silhouette(data,25),self.number_of_regions)
        
        # using agglomerative clustering to obtain segmentation 
        clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=self._n_cluster,affinity='cosine',linkage='average',distance_threshold=None).fit(data)
        self._clustering = clustering.labels_
        # building flat segmentation and then reshaping
        #self._segmentation=[ [ self._clustering[value-1]+1 for value in line ] for line in self._presegmentation ] 
        self._segmentation=numpy.asarray([self._clustering[value-1]+1 for line in self._presegmentation for value in line]).reshape(self._presegmentation.shape)
        end = time.process_time()
        print("clustering computed in {} seconds".format(end-begin), file=sys.stderr)

