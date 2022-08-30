from skimage import io, color, measure
from skimage.future import graph
from skimage.util import img_as_float, img_as_int
from skimage.color import rgb2gray

from sklearn.cluster import AgglomerativeClustering, KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from gensim.models import Word2Vec

# https://github.com/fjean/pymeanshift
from pymeanshift import segment
from cv2 import imread
from src.utils.node2vec.src import node2vec as nv
from src.helper import _update_color_features, _color_features, silhouette, _hog_channel_gradient, _get_bins, _normalized_hog

from numpy import asarray, unique, amax, copy, argwhere, zeros, where, amin
from scipy.spatial.distance import cosine
from networkx import contracted_nodes, connected_components, is_connected
from math import sqrt


# ---DEV---
from sklearn.decomposition import PCA

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

        # RGB image
        self._image_rgb = io.imread(self._path_to_image)

        # L*a*b* image
        self._image = io.imread(self._path_to_image)
        self._image = img_as_float(self._image)
        self._image_lab = (color.rgb2lab(self._image) + [0,128,128]) #// [1,1,1]

        self._preseg_method = kwargs.get("preseg_method", "MS")
        self._presegmentation = kwargs.get("presegmentation", None)
        self._embeddings = kwargs.get("embeddings", None)
        self._dimensions = kwargs.get("dimensions", 16)

        self._docontiguous = kwargs.get("contiguous", False)
        self._domerge = kwargs.get("merge", False)
        self._verbose = kwargs.get("verbose", False)

        self._hs = kwargs.get("spatial_radius", 7)
        self._hr = kwargs.get("spatial_range",4.5)
        self._M = kwargs.get("min_density",50)
        self._sigma = kwargs.get("sigma",125)
        self._thr_pixels = kwargs.get("thr_pixels",250)
        self._thr_cosine = kwargs.get("thr_cosine",0.998)
        self._number_of_regions = 0

        self._RAG = None
        self._merged_RAG = None
        self._feature_vector = None
        self._feature_vector_merge = None
        self._hog = None
        self._bins = None
        self._segmentation = None
        self._segmentation_merged = None
        self._clustering = None
        self._print = print if self._verbose else lambda *a, **k: None

        # if no presegmentation labels are provided 
        # TODO: offer different possibilities according to method
        if(self._presegmentation is None):
            self.compute_preseg()
            #self._n_cluster = min(self._n_cluster, self._number_of_regions)

        self._FV = asarray(_color_features(self._presegmentation,self._image_lab,None))
        # ---DEV--- use image_lab not image!!! IF BAD, USE img_as_float
        #self._hog = _hog_channel_gradient(rgb2gray(self._image),multichannel=False)
        self._hog = _hog_channel_gradient(self._image_lab,multichannel=True)
        #self._hog = _hog_channel_gradient(self._image_rgb,multichannel=True)
        self._bins = _get_bins(self._hog[0], self._hog[1], measure.regionprops(self._presegmentation,intensity_image=rgb2gray(self._image)))
        regions = measure.regionprops(self._presegmentation)

        # if no embeddings are provided
        if(self._embeddings is None):
            self.compute_embeddings(regions)

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
        self._print("presegmentation computed in {} seconds".format(end-begin), file=sys.stderr)

    def compute_embeddings(self,regions):
        """
        Compute the RAG and embeddings from the initial presegmentation
        """
        import time, sys
        
        # computing RAG
        # TODO: add exception/error if _presegmentation is None
        begin = time.process_time() 
        # ---DEV--- do not use rag_mean_color, just RAG, and then weight edges according to color AND texture (see whatever paper)
        #self._RAG = graph.RAG(self._image_lab, self._presegmentation, connectivity=2)
        self._RAG = graph.rag_mean_color(self._image_lab,self._presegmentation,connectivity=2,mode='similarity',sigma=self._sigma)
        # ---DEV--- try with unnormalized bins of each region (_get_bins)
        # ---DEV--- DO NOT INCLUDE HOG FOR CLUSTERING
        # weighting the graph
        '''def _findregion(R):
            for i in range(len(regions)):
                if regions[i].label == R:
                    return i

        for u,v in self._RAG.edges():
            Ru = regions[_findregion(u)]
            Rv = regions[_findregion(v)]
            HOGu = [self._bins[Ru.label-1][j][0]/self._bins[Ru.label-1][j][1] for j in range(9)]
            HOGv = [self._bins[Rv.label-1][j][0]/self._bins[Rv.label-1][j][1] for j in range(9)]
            tuv = (1-cosine(HOGu,HOGv))
            cuv = self._RAG[u][v]['weight']
            self._RAG[u][v]['weight'] = (0.4 * sqrt(tuv*cuv)) + (0.6 * cuv)
        # ---DEV---
        end = time.process_time()
        self._print("RAG computed in {} seconds".format(end-begin), file=sys.stderr)'''

        # computing embeddings
        begin = time.process_time() 
        Gn2v = nv.Graph(self._RAG, False, 2, .5)
        Gn2v.preprocess_transition_probs()
        walks = Gn2v.simulate_walks(20, 20)
        walks = [list(map(str, walk)) for walk in walks]
        # FIXME: allow parameterization
        model = Word2Vec(walks, vector_size=self._dimensions, window=5, min_count=0, sg=1, workers=4, epochs=1)

        begin = time.process_time() 
        representation = model.wv
        nodes=self._RAG.nodes()
        self._embeddings = [representation.get_vector(str(node)).tolist() for node in nodes]
        end = time.process_time()
        self._print("embeddings computed in {} seconds".format(end-begin), file=sys.stderr)

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

        # we construct the feature vector of L*a*b* space + HOG features
        for_cosine = [ [0]*6 for _ in range(max([region.label for region in regions])) ]
        for region in regions:
            # getting the percentage of orientation bins for every region
            FV = self._feature_vector_merge[region.label-1].tolist()
            #HOG = [self._bins[region.label-1][j][0]/self._bins[region.label-1][j][1] for j in range(9)]
            for_cosine[region.label-1] = FV#+HOG
        # ---DEV--- try with this
        for_cosine = normalize(asarray(for_cosine))

        for i in range(len(regions)):
            Ri = regions[i]
            lenRi = len(Ri.coords)
            if(lenRi < thr_pixels):
                neighbors = list(self._merged_RAG.neighbors(Ri.label))
                closest = max([(regions[_findregion(Rj)].label,1-cosine(for_cosine[Ri.label-1],for_cosine[regions[_findregion(Rj)].label-1])) for Rj in neighbors],key=lambda x: x[1])[0]
                #closest = max([(regions[_findregion(Rj)].label,self._merged_RAG[Ri.label][regions[_findregion(Rj)].label]['weight']) for Rj in neighbors], key=lambda x: x[1])[0]
                Rj = regions[_findregion(closest)]
                max_label = Ri if Ri.label > Rj.label else Rj
                min_label = Ri if Ri.label < Rj.label else Rj
                for (x,y) in max_label.coords:
                    self._segmentation_merged[(x,y)] = min_label.label
                self._merged_RAG = contracted_nodes(self._merged_RAG,min_label.label,max_label.label,self_loops=False)
                coords=regions[_findregion(min_label.label)].coords.tolist()
                coords.extend(regions[_findregion(max_label.label)].coords)
                self._feature_vector_merge[min_label.label-1] = asarray(_update_color_features(coords,self._image_lab,self._image_rgb))
                #self._feature_vector_merge[min_label.label-1] = (self._feature_vector_merge[min_label.label-1]+self._feature_vector_merge[max_label.label-1])/2
                #self._bins[min_label.label-1] = [(self._bins[min_label.label-1][j][0]+self._bins[max_label.label-1][j][0], self._bins[min_label.label-1][j][1]+self._bins[max_label.label-1][j][1]) for j in range(9)]
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

        # we construct the feature vector of L*a*b* space + HOG features
        for_cosine = [ [0]*6 for _ in range(max([region.label for region in regions])) ]
        for region in regions:
            # getting the percentage of orientation bins for every region
            FV = self._feature_vector_merge[region.label-1].tolist()
            #HOG = [self._bins[region.label-1][j][0]/self._bins[region.label-1][j][1] for j in range(9)]
            for_cosine[region.label-1] = FV#+HOG
        # ---DEV--- try with this first
        for_cosine = normalize(asarray(for_cosine))

        for u,v in self._merged_RAG.edges():
            Ri=regions[_findregion(u)]
            Rj=regions[_findregion(v)]
            sim=1-cosine(for_cosine[Ri.label-1],for_cosine[Rj.label-1])
            if sim >= thr:
                max_label = Ri if Ri.label > Rj.label else Rj
                min_label = Ri if Ri.label < Rj.label else Rj
                for (x,y) in max_label.coords:
                    self._segmentation_merged[(x,y)] = min_label.label
                # updating remaining labels
                #_, labels_merge = unique(labels_merge,return_inverse=1)
                #labels_merge=(1+labels_merge).reshape(labels.shape)
                # updating feature vector
                self._merged_RAG = contracted_nodes(self._merged_RAG,min_label.label,max_label.label,self_loops=False)
                # UPDATE --- previous computation was just a proxy: if this is better, 
                #        --- define a function to update ONE region only
                # getting pixels' coordinates of merged region to update feature vector
                coords=regions[_findregion(min_label.label)].coords.tolist()
                coords.extend(regions[_findregion(max_label.label)].coords)
                self._feature_vector_merge[min_label.label-1] = asarray(_update_color_features(coords,self._image_lab,self._image_rgb))
                #self._feature_vector_merge[min_label.label-1] = (self._feature_vector_merge[min_label.label-1]+self._feature_vector_merge[max_label.label-1])/2
                #self._bins[min_label.label-1] = [(self._bins[min_label.label-1][j][0]+self._bins[max_label.label-1][j][0], self._bins[min_label.label-1][j][1]+self._bins[max_label.label-1][j][1]) for j in range(9)]
                return True
        return False

        '''for u,v in self._merged_RAG.edges():
            Ri=regions[_findregion(u)]
            Rj=regions[_findregion(v)]
            sim = self._merged_RAG[u][v]['weight']
            if sim >= thr:
                max_label = Ri if Ri.label > Rj.label else Rj
                min_label = Ri if Ri.label < Rj.label else Rj
                for (x,y) in max_label.coords:
                    self._segmentation_merged[(x,y)] = min_label.label
                self._merged_RAG = contracted_nodes(self._merged_RAG,min_label.label,max_label.label,self_loops=False)
                return True
        return False'''

    # ---DEV---
    # custom weights for graph
    # ---DEV---

    def _merge(self,thr_pixels=750,thr=0.995):
        """
        (Private) Procedure that merge while possible. First pixels, then similarity. 
        This is Algorithm 2 from GeSt: a new image segmentation technique based on graph embedding.
        
        :param thr_pixels:
            The threshold size for merging.
        :param thr:
            The threshold for merging. This value depends on the distance considered. 
        """

        if(self._segmentation_merged is None):
            self._segmentation_merged = copy(self._segmentation)
        # initial computation, will be maintained during algorithm
        # ---DEV--- try different graphs and different parameters
        # ---DEV--- add a function to compute a graph with custom weights!
        # ---DEV--- feature_vector must be an attribute, computed once and then updated!
        self._merged_RAG = graph.RAG(self._segmentation_merged,connectivity=1)
        #self._merged_RAG = graph.rag_mean_color(self._image_lab,self._segmentation_merged,connectivity=2,mode='similarity',sigma=self._sigma)

        # initializing feature vector --- L*a*b* space
        self._feature_vector_merge = asarray(_color_features(self._segmentation_merged,self._image_lab,self._image_rgb))
        # ---DEV--- use hog on grayscale here? I DO NOT REALLY SEE WHY...
        self._bins = asarray(_get_bins(self._hog[0],self._hog[1],measure.regionprops(self._segmentation_merged)))
        #self._bins = asarray(_get_bins(measure.regionprops(self._segmentation_merged,intensity_image=rgb2gray(self._image))))

        while(True):
            # ---DEV--- if this does not work properly, then relabel regions AT EACH STEP (see ---CMT---)
            regions = measure.regionprops(self._segmentation_merged)
            merged = self._similarity_merge(regions,thr)
            if(merged):
                continue
            merged = self._pixels_merge(regions,thr_pixels)
            if(merged):
                continue
            break

        # ---CMT--- this assures that labels are contiguous from 1 to N!
        # ---CMT--- we get the *indices* of the unique values, 
        _, self._segmentation_merged = unique(self._segmentation_merged,return_inverse=1)
        self._segmentation_merged=(1+self._segmentation_merged).reshape(self._presegmentation.shape)

    # TODO : verbose print!
    def segmentation(self):
        """
        Member method that implements Algorithm 1 of the paper GeSt: a new image segmentation technique based on graph embedding.
        """

        import time, sys
        import matplotlib.pyplot as plt

        # NOTE: Mean is included in graph somehow?
        begin = time.process_time() 
        # ---DEV--- 
        # to try with just feature vector
        # self._embeddings = [ list() for _ in range(len(self._feature_vector)) ]
        # ---DEV---
        for l,v in enumerate(self._feature_vector):
            # --- FV ONLY ---
            self._embeddings[l].extend(v)
            # --- FV+HOG ---
            # ----DEV--- add also RGB HOG? (try with only this one first ?!)
            #self._embeddings[l].extend([self._bins[l][j][0]/self._bins[l][j][1] for j in range(9)])
        end = time.process_time()
        self._print("feature vector computed in {} seconds".format(end-begin), file=sys.stderr)

        # clustering
        begin = time.process_time() 
        scaler = StandardScaler()
        data = scaler.fit_transform(self._embeddings)

        '''pca = PCA()
        data = pca.fit_transform(data)
        pca_variance = pca.explained_variance_
        plt.figure(figsize=(8, 6))
        plt.bar(range(pca_variance.shape[0]), pca_variance, alpha=0.5, align='center', label='individual variance')
        plt.legend()
        plt.ylabel('Variance ratio')
        plt.xlabel('Principal components')
        plt.show()'''
            
        # we keep only eigenvalues > 1: Kaiser's criterion
        '''pca = PCA()
        data = pca.fit_transform(data)
        pca_variance = pca.explained_variance_
        n_components = len([e for e in pca.explained_variance_ if e >= 1])
        pca = PCA(n_components=n_components)
        data = pca.fit_transform(data)'''

        if(self._n_cluster is None):
            self._n_cluster = min(silhouette(data,25),self._number_of_regions)
        
        # using agglomerative clustering to obtain segmentation 
        # ---DEV--- bringing back k-means!
        #clustering = KMeans(init = "k-means++", n_clusters = self._n_cluster, n_init = 35, random_state=10)
        clustering = AgglomerativeClustering(n_clusters=self._n_cluster,affinity='cosine',linkage='average',distance_threshold=None)
        clustering.fit(data)
        self._clustering = clustering.labels_
        end = time.process_time()
        self._print("clustering computed in {} seconds".format(end-begin), file=sys.stderr)

        # building flat segmentation and then reshaping
        self._segmentation=asarray([self._clustering[value-1]+1 for line in self._presegmentation for value in line]).reshape(self._presegmentation.shape)

        if(self._docontiguous):
            self._contiguous()
        if(self._domerge):
            self._merge(thr_pixels=self._thr_pixels,thr=self._thr_cosine)
            #self.dev_merge(thr_pixels=self._thr_pixels,thr=self._thr_cosine)
        self._number_of_regions = len(unique(self._segmentation))
        self._print("final segmentation has {} regions".format(self._number_of_regions))

    def dev_merge(self,thr_pixels=250,thr=0.998,sigma=125):
        import time, sys
        begin = time.process_time()
        self.merge_pixels(thr_pixels)
        self.merge_cosine(thr)
        end = time.process_time()
        self._print("merging procedure done in {} seconds".format(end-begin))

    def compare(self,path_groundtruths,filename):
        from src.helper import _get_groundtruth
        from src.measures import _probabilistic_rand_index
        gt_boundaries, gt_segmentation = _get_groundtruth(path_groundtruths+filename[:-4]+".mat")
        pri = _probabilistic_rand_index(gt_segmentation,self._segmentation)
        if(self._domerge):
            pri_merged = _probabilistic_rand_index(gt_segmentation,self._segmentation_merged)
        else:
            pri_merged = 0
        return pri, pri_merged
