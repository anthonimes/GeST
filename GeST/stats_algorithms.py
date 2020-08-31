import numpy
from statistics import mean,stdev
from utils.parse_matlab import get_groundtruth, get_BSR
from os import walk, environ
import argparse, pickle
from utils.metrics.pri import probabilistic_rand_index, rand_index_score, f_measure
from skimage.metrics import adapted_rand_error


def _best_worst_groundtruth_FMEASURE(GT,BSR):
    # we compure the ARI between every pair of path_groundtruths
    #ri = rand_index_score(GT[i].flatten().tolist(),GT[j].flatten().tolist())
    for gt in GT:
        tmp=[]
        tmp.append(f_measure(gt.flatten().tolist(),BSR.flatten().tolist()))
    return max(tmp)
    # the best groundtruth is the one maximizing sum(row) or sum(column) --- i.e. best PRI
    # the worst groundtruth is the one minimizing sum(row) or sum(column) --- i.e. worst PRI

def _best_worst_groundtruth_PRI(GT,BSR):
    return probabilistic_rand_index(GT,BSR)

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
    path_BSR = argsy['path']+"/BSR/"
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
    path_BSR = path_BSR+"val_labels/"
    path_groundtruths = path_groundtruth+folder
    best = True if argsy['best'] == "True" else False

    if method == "SLIC":
        common=method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_graphs = "results/graphs/"+common
        path_pickles = "results/pickles/"+common
        path_labels = "results/labels/"+common
        path_scores = "results/scores/"+common
        path_figs = "results/figs/"+common
        path_presegs = "results/presegs/"+common
        path_embeddings = "results/embeddings/"+common
        path_clusterings = "results/clusterings/"+common
        path_matlab = "results/matlab/"+common
    else:
        common=method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/"+folder
        path_graphs = "results/graphs/"+common
        path_pickles = "results/pickles/"+common
        path_labels = "results/labels/"+common
        path_scores = "results/scores/"+common
        path_figs = "results/figs/"+common
        path_presegs = "results/presegs/"+common
        path_embeddings = "results/embeddings/"+common
        path_clusterings = "results/clusterings/"+common
        path_matlab = "results/matlab/"+common

    if argsy['best'] == "True":
        path_graphs+="best/"
        path_pickles+="best/"
        path_labels+="best/"
        path_figs+="best/"
        path_presegs+="best/"
        path_embeddings+="best/"
        path_clusterings+="best/"
        path_matlab+="best/"

    PRI_BSD, PRI_GEST, FMEASURE_GEST, FMEASURE_BSD = [], [], [], []
    # load the image and convert it to a floating point data type
    for (dirpath, dirnames, filenames) in walk(path_images):
        for i,filename in enumerate(sorted(filenames)):
            if filename.endswith(".jpg"):
                gt_boundaries, gt_segmentation = get_groundtruth(path_groundtruths+filename[:-4]+".mat")
                print("BSD {}: {}".format(i+1,filename[:-4]),end=' ')
                gt_BSD = get_BSR(path_BSR+filename[:-4]+".mat")
                pri =_best_worst_groundtruth_PRI(gt_segmentation,gt_BSD)
                fm =_best_worst_groundtruth_FMEASURE(gt_segmentation,gt_BSD)
                print(pri,fm)
                PRI_BSD.append(pri)
                FMEASURE_BSD.append(fm)
                gt_GEST = pickle.load(open(path_labels+str(i+1)+"_"+filename[:-4]+".seg","rb"))
                pri =_best_worst_groundtruth_PRI(gt_segmentation,gt_GEST)
                fm =_best_worst_groundtruth_FMEASURE(gt_segmentation,gt_GEST)
                print("GEST {}: {}".format(i+1,filename[:-4]),end=' ')
                print(pri,fm)
                PRI_GEST.append(pri)
                FMEASURE_GEST.append(fm)
    print("MEAN PRI BSD, MEAN FMEASURE BSD\t {} {}".format(mean(PRI_BSD),mean(FMEASURE_BSD)))
    print("MEAN PRI GEST, MEAN FMEASURE GEST\t {} {}".format(mean(PRI_GEST),mean(FMEASURE_GEST)))
