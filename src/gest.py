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
from src.helper import _color_features, silhouette, _colors, _colors_by_region

from numpy import asarray, unique, copy, amax, copy, argwhere, zeros
from scipy.spatial.distance import cosine
from networkx import contracted_nodes, draw_networkx, connected_components, is_connected

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
        self._segmentation_merged = None
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
        model = Word2Vec(walks, size=16, window=5, min_count=0, sg=1, workers=4, iter=1)

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
        # building flat segmentation and then reshaping
        #self._segmentation=[ [ self._clustering[value-1]+1 for value in line ] for line in self._presegmentation ] 
        self._segmentation=asarray([self._clustering[value-1]+1 for line in self._presegmentation for value in line]).reshape(self._presegmentation.shape)
        end = time.process_time()
        print("clustering computed in {} seconds".format(end-begin), file=sys.stderr)
        print("final segmentation has {} regions".format(amax(self._segmentation)))

    # small regions merging --- noise removal
    def merge_pixels(self,thr_pixels=100,sigma=125):
        import time, sys
        # NOTE; labels must be a matrix-like imaeg
        begin = time.process_time()
        if(self._segmentation_merged is None):
            self._segmentation_merged = copy(self._segmentation)
        merged=True
        # initial computation, will be maintained during algorithm
        G = graph.RAG(self._segmentation_merged,connectivity=2)
        regions = measure.regionprops(self._segmentation_merged)
            
        def _findregion(R):
            for i in range(len(regions)):
                if regions[i].label == R:
                    return regions[i]

        for i in range(len(regions)):
            Ri = regions[i]
            if(Ri.label in G.nodes()):
                lenRi = len(Ri.coords)
                if(lenRi < thr_pixels):
                    merged=True
                    # WARNING: neighbors in graphs are labels, not indices of regions array!
                    neighbors = list(G.neighbors(Ri.label))
                    closest = max([(_findregion(Rj).label,1-cosine(self._FV[Ri.label-1],self._FV[_findregion(Rj).label-1])) for Rj in neighbors],key=lambda x: x[1])[0]
                    Rj = _findregion(closest)
                    R_max_label = Ri if Ri.label > Rj.label else Rj
                    R_min_label = Ri if Ri.label < Rj.label else Rj
                    for (x,y) in R_max_label.coords:
                        self._segmentation_merged[(x,y)] = R_min_label.label
                    # updating feature vector
                    self._FV[R_min_label.label-1] = (self._FV[R_min_label.label-1]+self._FV[R_max_label.label-1])/2
                    # merging nodes in the RAG
                    G = contracted_nodes(G,R_min_label.label,R_max_label.label,self_loops=False)
            
        _, self._segmentation_merged = unique(self._segmentation_merged,return_inverse=1)
        self._segmentation_merged=(1+self._segmentation_merged).reshape(self._segmentation.shape)
        end = time.process_time()
        print("merging procedure done in {} seconds".format(end-begin))
        print("final segmentation has {} regions".format(amax(self._segmentation_merged)))

    def contiguous(self):
        import time, sys
        import matplotlib.pyplot as plt
        Gr = graph.RAG(self._presegmentation, connectivity=1)
        begin = time.process_time()
        # MERGING CONTIGUOUS REGIONS ONLY IN A FIRST PLACE
        new_labels_clustering = copy(self._clustering)
        for _label in unique(self._clustering):
            labelmax = amax(new_labels_clustering)
            # getting regions with this label
            vertices = 1+argwhere(self._clustering == _label).flatten()
            # ugly but if connected, the if will fail...
            Gc = Gr if len(Gr.subgraph(vertices).edges())==0 else Gr.subgraph(vertices)
            connected_component = sorted(connected_components(Gc), key=len, reverse=True)
            if(len(connected_component)>1):
                to_relabel=connected_component[1:]
                labelcpt=1
                for cc in to_relabel:
                    for vertex in cc:
                        new_labels_clustering[vertex-1]=labelmax+labelcpt
                    labelcpt+=1

        # computing corresponding new segmentation
        for l,line in enumerate(self._presegmentation):
            for j,value in enumerate(line):
                self._segmentation[l][j] = new_labels_clustering[value-1]+1
        
        end = time.process_time()
        print("contiguous done in {} seconds".format(end-begin))
        print("final segmentation has {} regions".format(len(unique(self._segmentation))))
        '''fig, ax = plt.subplots()
        colored_regions = color.label2rgb(self._segmentation, self._image, alpha=1, colors=_colors_by_region(amax(self._segmentation)), bg_label=0)
        im = ax.imshow(colored_regions)

        regions = measure.regionprops(self._segmentation)
        for region in regions:
            xy = region.centroid
            x = xy[1]
            y = xy[0]
            text = ax.text(x, y, region.label,ha="center", va="center", color="w")

        plt.show()'''

    def merge_cosine(self,thr=0.997,sigma=125):
        import matplotlib.pyplot as plt
        import time, sys
        # NOTE; labels must be a matrix-like imaeg
        begin = time.process_time()
        if(self._segmentation_merged is None):
            self._segmentation_merged = copy(self._segmentation)
        merged=True
        # initial computation, will be maintained during algorithm
        G = graph.RAG(self._segmentation_merged,connectivity=1)

        def _findregion(R):
            for i in range(len(regions)):
                if regions[i].label == R:
                    return regions[i]

        #while(merged):
        regions = measure.regionprops(self._segmentation_merged)
        # FIXME: useless to compute again the ones that have not changed
        merged=False
        regions_merged = [False]*(len(regions)+1)
    
        # similarity merging
        for u,v in G.edges():
            Ri=_findregion(u)
            Rj=_findregion(v)
            if not(regions_merged[Ri.label] or regions_merged[Rj.label]):
                sim=1-cosine(self._FV[Ri.label-1],self._FV[Rj.label-1])
                if sim >= thr:
                    print("merging {} and {}".format(Ri.label,Rj.label))
                    merged=True
                    #print("similarity merging region {} and {}.".format(Ri.label,Rj.label))
                    R_max_label = Ri if Ri.label > Rj.label else Rj
                    R_min_label = Ri if Ri.label < Rj.label else Rj
                    regions_merged[R_max_label.label]=True
                    regions_merged[R_min_label.label]=True
                    for (x,y) in R_max_label.coords:
                        self._segmentation_merged[(x,y)] = R_min_label.label
                    # updating feature vector
                    self._FV[R_min_label.label-1] = (self._FV[R_min_label.label-1]+self._FV[R_max_label.label-1])/2
                    # merging nodes in the RAG
                    G = contracted_nodes(G,R_min_label.label,R_max_label.label,self_loops=False)
            #if(merged):
            #    break
            
        _, self._segmentation_merged = unique(self._segmentation_merged,return_inverse=1)
        self._segmentation_merged=(1+self._segmentation_merged).reshape(self._segmentation.shape)
        end = time.process_time()
        print("merging procedure done in {} seconds".format(end-begin))
        print("final segmentation has {} regions".format(amax(self._segmentation_merged)))

    def all_merge(self,thr_pixels=250,thr=0.998,sigma=5):
        import time, sys
        # NOTE; labels must be a matrix-like imaeg
        begin = time.process_time()
        merged=True
        if(self._segmentation_merged is None):
            self._segmentation_merged = copy(self._segmentation)
        # initial computation, will be maintained during algorithm
        G = graph.RAG(self._segmentation_merged,connectivity=1)

        def _findregion(R):
            for i in range(len(regions)):
                if regions[i].label == R:
                    return i

        while(merged):
            regions = measure.regionprops(self._segmentation_merged)
            # FIXME: totally useless to compute again the ones that have not changed
            merged=False

            for u,v in G.edges():
                Ri=regions[_findregion(u)]
                Rj=regions[_findregion(v)]
                sim=1-cosine(self._FV[Ri.label-1],self._FV[Rj.label-1])
                if sim >= thr:
                    #print("similarity merging region {} and {}.".format(Ri.label,Rj.label))
                    max_label = Ri if Ri.label > Rj.label else Rj
                    min_label = Ri if Ri.label < Rj.label else Rj
                    for (x,y) in max_label.coords:
                        self._segmentation_merged[(x,y)] = min_label.label
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
                        self._segmentation_merged[(x,y)] = min_label.label
                    merged=True
                    has_merged=True
                    # updating feature vector
                    self._FV[min_label.label-1] = (self._FV[min_label.label-1]+self._FV[max_label.label-1])/2
                    G = contracted_nodes(G,min_label.label,max_label.label,self_loops=False)
                if(merged):
                    break
            if(merged):
                continue

        _, self._segmentation_merged = unique(self._segmentation_merged,return_inverse=1)
        self._segmentation_merged=(1+self._segmentation_merged).reshape(self._presegmentation.shape)
        end = time.process_time()
        print("merging procedure done in {} seconds".format(end-begin))

    '''def merge(self,thr_pixels=200,thr=0.995,sigma=5):
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
                continue'''

    def merge(self,thr_pixels=250,thr=0.998,sigma=125):
        self.contiguous()
        self.merge_pixels(thr_pixels)
        #self.merge_cosine(thr)

