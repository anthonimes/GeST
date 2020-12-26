# GeST: Graph embedding Segmentation Technique
  
This repository provides a reference implementation for `GeST`, an image segmentation technique based on graph embedding as presented in:
> GeSt: a new image segmentation technique based on graph embedding
> A. Perez
> Accepted for publication at [VISAPP 2021](http://www.visapp.visigrapp.org/presentationdetails.aspx)

## How to use

`src/gest.py` provides a class for computing segmentations. The objects need to be given 
initial embeddings and presegmentation to compute the final segmentation. If `None` is given on any 
of these parameters, default algorithms (namely `MeanShift` and `node2vec`) will be used to compute them.
A script is given as example in `main.py` and can be used as follows:

`python3 main.py -p examples`

The only required argument consists in a path to folder containing images to segment.
Other arguments optional can be provided and are described using the `-h` option.

```
```

### Requirements

+ `python >= 3.6`
+ `scikit-learn >= ??`
+ `scikit-image` 
+ `numpy`
+ `pymeanshift` --- https://github.com/fjean/pymeanshift
+ `node2vec` --- https://github.com/aditya-grover/node2vec

## Examples of application

### With default parameters

![Default parameters](/images/gestapp.png)

### With custom parameters

