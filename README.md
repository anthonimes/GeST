# GeST: Graph embedding Segmentation Technique

This repository provides a reference implementation for `GeST`, an image segmentation technique based on graph embedding as presented in:  
> GeSt: a new image segmentation technique based on graph embedding
> A. Perez
> Accepted for publication at [http://www.visapp.visigrapp.org/presentationdetails.aspx](VISAPP 2021)

## How to use

The `src/gest.py` file provides a class for computing segmentations. The objects need to be given 
initial embeddings and presegmentation to compute the final segmentation. If `None` is given on any 
of these parameters, default algorithms will be used to compute them. 

### Requirements

## Examples of application

### With default parameters

![Default parameters](/images/gestapp.png)

### With custom parameters
