"""
main.py: example for using GeST
A description of required and optional arguments can be found using python3 main.py -h
For completeness, the output of this command is provided below: 
    usage: main.py [-h] -p PATH [-m METHOD] [--sigma SIGMA] [-n NCLUSTERS] [--silhouette] [--hs HS] [--hr HR] [--mind MIND] [--merge] [--contiguous] [-s]

    optional arguments:
      -h, --help            show this help message and exit
      -p PATH, --path PATH  path to folder containing images (default: None)
      -m METHOD, --method METHOD
                            pre-segmentation method (default: msp)
      --sigma SIGMA         kernel parameter (default: 125)
      -n NCLUSTERS, --nclusters NCLUSTERS
                            number of clusters (default: 21)
      --silhouette          use silhouette method instead of fixed number of clusters (default: False)
      --hs HS               spatial radius (default: 7)
      --hr HR               range radius (default: 4.5)
      --mind MIND           min density (default: 50)
      --merge               apply merging procedure (default: False)
      --contiguous          compute contiguous regions (default: False)
      -s, --save            save files to hard drive (default: False)
"""
from os import walk
from src.helper import _parse_args, _write, _display
from src.gest import GeST

import warnings
warnings.filterwarnings("ignore")

arguments = _parse_args()

dirpath,_,images = list(walk(arguments['path']))[0]

path_merge = "/".join(arguments['path'].split("/")[:-2])
_, dirnames, hardimages = list(walk(path_merge+"/from_observation"))[0]

# ---DEV--- in order to parallelize image treatments
from multiprocessing import Pool
from statistics import mean
from sys import stderr

class Segment(object):
    def __init__(self,dirpath,n_cluster,parameters):
        # dict of parameters
        self.parameters = parameters
        self.dirpath = dirpath
        self.n_cluster = n_cluster
    def __call__(self, filename):
            g = GeST(self.dirpath+filename, self.n_cluster, **self.parameters)
            g.segmentation()
            # groundtruth comparison
            pri, pri_merged = g.compare(arguments['groundtruth'],filename)
            # ---DEV--- make the segmentation contiguous before computing PRI!
            # do the segmentation(_merged) RAG, then for each label check for connectivity 
            # (BUT checking if the graph is not null before: should not happen if we work on proper labels 
            '''g._contiguous_merge()
            pri_contiguous,pri_contiguous_merged = g.compare(arguments['groundtruth'],filename)'''
            return (filename,pri,pri_merged)#,pri_contiguous,pri_contiguous_merged)

if(arguments['contiguous']):
    # ---DEV--- merging small regions on contiguous seems bad!
    pixels=[0]#50,150,250,350]
else:
    pixels=[750]#250,750,1000,1250]

cosine=[0.998]
dimensions = [16,32]

for d in dimensions:
    for thr_pixels in pixels:
        for thr_cosine in cosine:
            PRI,PRI_CONTIGUOUS,PRI_MERGED,PRI_CONTIGUOUS_MERGED,BEST_PRI,SELECTED_PRI  = [], [], [], [], [], []
            for i in range(1):
                # ---DEV--- replace by lists (but OK since small dataset)
                best_pri, selected_pri, mean_pri, mean_pri_merged, mean_pri_contiguous, mean_pri_contiguous_merged = dict(), dict(), dict(), dict(), dict(), dict()
                #for i,filename in enumerate(sorted(images)):
                #print("===== processing image {}: {}".format(i,filename),end=' ')
                # ---DEV--- dict of arguments
                kwargs = {
                          "preseg_method": arguments['method'], \
                          "merge": arguments['merge'], \
                          "contiguous": arguments['contiguous'], \
                          "verbose": arguments['verbose'], \
                          "thr_pixels": thr_pixels, \
                          "dimensions": d, \
                          "thr_cosine": thr_cosine
                          }

                '''try:
                    pool = Pool() # remove number to use all
                    segment = Segment(dirpath,arguments['n_cluster'],kwargs)
                    results = pool.map(segment, sorted(images))
                    # ---DEV--- what does the result look like?
                    for filename,pri,pri_merged in results:
                        mean_pri[filename] = pri
                        mean_pri_merged[filename] = pri_merged
                        #mean_pri_contiguous[filename] = pri_contiguous
                        if(pri_merged>pri):
                            best_pri[filename] = pri_merged
                        else:
                            best_pri[filename] = pri
                        if (filename in hardimages):
                            selected_pri[filename] = pri_merged
                        else:
                            selected_pri[filename] = pri
                finally: # To make sure processes are closed in the end, even if errors happen
                    pool.close()
                    pool.join()'''

                for filename in sorted(images):
                    #g = GeST(dirpath+images[0], arguments['n_cluster'], **kwargs)
                    #g.segmentation()
                    g = GeST(dirpath+filename, arguments['n_cluster'], **kwargs)
                    g.segmentation()
                    # groundtruth comparison
                    pri, pri_merged = g.compare(arguments['groundtruth'],filename)

                    if(arguments['groundtruth'] != ""):
                        mean_pri[filename] = pri
                        mean_pri_merged[filename] = pri_merged
                        if(pri_merged>pri):
                            best_pri[filename] = pri_merged
                        else:
                            best_pri[filename] = pri
                        if (filename in hardimages):
                            selected_pri[filename] = pri_merged
                        else:
                            selected_pri[filename] = pri
                    #print("iteration {}/{}/{} done: classical {}, selected {}, best {}".format(d,thr_pixels,thr_cosine,mean(mean_pri.values()),mean(selected_pri.values()), mean(best_pri.values())), file=stderr)

                PRI.append(mean(mean_pri.values()))
                #PRI_CONTIGUOUS.append(mean(mean_pri_contiguous.values()))
                PRI_MERGED.append(mean(mean_pri_merged.values()))
                #PRI_CONTIGUOUS_MERGED.append(mean(mean_pri_contiguous_merged.values()))
                BEST_PRI.append(mean(best_pri.values()))
                SELECTED_PRI.append(mean(selected_pri.values()))

            print("===== ROUND: {} and {} =====".format(thr_pixels, thr_cosine))
            print("Mean Probabilistic Rand Index:",mean(PRI))
            #print("Mean Contiguous Probabilistic Rand Index:",mean(PRI_CONTIGUOUS))
            print("Mean Merged Probabilistic Rand Index:",mean(PRI_MERGED))
            #print("Mean Contiguous Merged Probabilistic Rand Index:",mean(PRI_CONTIGUOUS_MERGED))
            print("Mean Best Probabilistic Rand Index:",mean(BEST_PRI))
            print("Mean Selected Probabilistic Rand Index:",mean(SELECTED_PRI))
