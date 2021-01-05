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

for i,filename in enumerate(sorted(images)):
    print("===== processing image {}".format(filename))

    g = GeST(dirpath+filename, arguments['n_cluster'], preseg_method=arguments['method'],merge=arguments['merge'],contiguous=arguments['contiguous'])
    g.segmentation()
    
    # writing to hard drive or displaying with matplotlib
    if(arguments['save']): 
        _write(g,arguments)
    else:
        _display(g,arguments)
