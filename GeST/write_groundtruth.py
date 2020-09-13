import numpy
from skimage.util import img_as_float
from statistics import mean,stdev
from os import walk, makedirs
import argparse, helper
from skimage import io
from skimage.metrics import adapted_rand_error

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = False, help = "Path to the image")
    ap.add_argument("-p", "--path", required = False, help = "Path to folder")
    ap.add_argument("-m", "--method", required = True, help="pre-segmentation method")
    ap.add_argument("-b", "--best", required = False, help="compute best clustering?", default=False)
    ap.add_argument("-w", "--write", required = False, help="write all files to hard drive?", default=False)
    ap.add_argument("-d", "--dataset", required = False, help="which of {train,val,test} to evaluate?", default="val")
    ap.add_argument("--hs", required = False, help="spatial radius?", default=15)
    ap.add_argument("--hr", required = False, help="range radius?", default=4.5)
    ap.add_argument( "--mind", required = False, help="min density", default=300)
    ap.add_argument( "--sigma", required = True, help="kernel parameter", default=50)
    ap.add_argument( "--segments", required = False, help="number of segments (SLIC)", default=50)
    ap.add_argument( "--compactness", required = False, help="compactness (SLIC)", default=50)

    argsy = vars(ap.parse_args())
    path_image = argsy['path']+"/images/"
    path_groundtruth = argsy['path']+"/groundTruth/"

    path_results = "results/"
    _spatial_radius=int(argsy['hs']) #hs
    _range_radius=float(argsy['hr']) #hr
    _min_density=int(argsy['mind'])
    _sigma=float(argsy['sigma'])
    _num_segments = float(argsy['segments'])
    _compactness = float(argsy['compactness'])

    methods = { "slic": "SLIC", "msp": "MSP", "mso": "MSO" }
    method = methods[argsy['method']]

    which_folder = {"val": "val/", "train": "train/", "test": "test/"}
    folder = which_folder[argsy['dataset']]

    path_images = argsy['path']+"/images/"+folder
    path_groundtruths = path_groundtruth+folder
    path_write_gt = "results/groundtruth/"+method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder
    
    makedirs(path_write_gt, exist_ok=True)

    if method == "SLIC":
        path_pickles = "results/pickles/"+method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_labels = "results/labels/"+method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_embeddings = "results/embeddings/"+method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_clusterings = "results/clusterings/"+method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"+folder
        name = method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)
    else:
        path_pickles = "results/pickles/"+method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_labels = "results/labels/"+method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_embeddings = "results/embeddings/"+method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_clusterings = "results/clusterings/"+method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder
        name = method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)

    # computing best clustering ?
    argsy['best'] = True if argsy['best'] == "True" else False
    if argsy['best'] is True:
        path_pickles+="best/"
        path_labels+="best/"
        path_embeddings+="best/"
        path_clusterings+="best/"

    PRI, SEGMENTS, MEAN_PRI, FMEASURE, MEAN_FMEASURE = [], [], [], [], []
    # load the image and convert it to a floating point data type
    for (dirpath, dirnames, filenames) in walk(path_images):
        for i,filename in enumerate(sorted(filenames)):
            if filename.endswith(".jpg"):
                print("{}: {}".format(i+1,filename[:-4]))
                image = io.imread(dirpath+filename)
                image = img_as_float(image)
                gt_boundaries, gt_segmentation = helper._get_groundtruth(path_groundtruths+filename[:-4]+".mat")
                for i in range(len(gt_segmentation)):
                    helper._savefig(gt_segmentation[i], image, path_write_gt+filename[:-4]+"_"+str(i)+"_"+str(len(numpy.unique(gt_segmentation[i])))+".png")
