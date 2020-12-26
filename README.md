# GeST: Graph embedding Segmentation Technique
  
This repository provides a reference implementation for `GeST`, an image segmentation technique based on graph embedding as presented in:
> GeSt: a new image segmentation technique based on graph embedding
> A. Perez
> Accepted for publication at [VISAPP 2021](http://www.visapp.visigrapp.org/presentationdetails.aspx)

## How to use

`GeST` uses two modules as subroutines. Once cloned, the repository needs to be prepared using: 

```
git submodule init
git submodule update
```

Submodules are used in order to keep track of possible modifications. However, `node2vec` is not 
compatible with `python3` so far, and hence one needs to remove lines `46` and `48` of the following file: 

`src/utils/node2vec/src/node2vec.py`

`src/gest.py` provides a class for computing segmentations. The objects need to be given 
initial embeddings and presegmentation to compute the final segmentation. If `None` is given on any 
of these parameters, default algorithms (namely `MeanShift` and `node2vec`) will be used to compute them.
A script is given as example in `main.py` and can be used as follows:

`python3 main.py -p examples`

The only required argument consists in a path to folder containing images to segment.
Other arguments optional can be provided and are described using the `-h` option.

```
usage: main.py [-h] -p PATH [-m METHOD] [--sigma SIGMA] [-n NCLUSTERS] [--silhouette] [--hs HS] [--hr HR] [--mind MIND] [--merge] [-w]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  path to folder containing images (default: None)
  -m METHOD, --method METHOD
                        pre-segmentation method (default: msp)
  --sigma SIGMA         kernel parameter (default: 125)
  -n NCLUSTERS, --nclusters NCLUSTERS
                        number of clusters (default: 24)
  --silhouette          use silhouette method instead of fixed number of clusters (default: False)
  --hs HS               spatial radius (default: 7)
  --hr HR               range radius (default: 4.5)
  --mind MIND           min density (default: 50)
  --merge               apply merging procedure (default: False)
  -w, --write           write all files to hard drive (default: False)
```

### Requirements

+ `python >= 3.6`
+ `scikit-learn >= ??`
+ `scikit-image` 
+ `numpy`
+ `pymeanshift` --- https://github.com/fjean/pymeanshift

## Examples of application

### With default parameters

![Default parameters](/images/gestapp.png)

### With custom parameters

