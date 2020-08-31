import numpy
from statistics import mean,stdev
from os import walk, environ
import argparse, helper
from skimage.metrics import adapted_rand_error


def _best_worst_groundtruth_FMEASURE(GT):
    matrice=numpy.zeros((len(GT),len(GT)))
    segments = [numpy.amax(gt) for gt in GT]
    positions = [(i,j) for i in range(len(GT)) for j in range(len(GT)) if i != j]
    for i,j in positions:
        # we compure the ARI between every pair of path_groundtruths
        #ri = rand_index_score(GT[i].flatten().tolist(),GT[j].flatten().tolist())
        fmeasure = helper._f_measure(GT[i].flatten().tolist(),GT[j].flatten().tolist())
        matrice[i,j]=fmeasure
    # the best groundtruth is the one maximizing sum(row) or sum(column) --- i.e. best PRI
    # the worst groundtruth is the one minimizing sum(row) or sum(column) --- i.e. worst PRI
    matrice=matrice[matrice>0].reshape((len(GT),len(GT)-1))
    best_FMEASURE = sum(matrice[matrice.sum(axis=1).argmax()])/matrice.shape[1]
    worst_FMEASURE = sum(matrice[matrice.sum(axis=1).argmin()])/matrice.shape[1]
    mean_FMEASURE = numpy.mean(matrice)
    return (matrice.sum(axis=1).argmax(), matrice.sum(axis=1).argmin(), best_FMEASURE, worst_FMEASURE, mean_FMEASURE, segments)

def _best_worst_groundtruth_PRI(GT):
    matrice=numpy.zeros((len(GT),len(GT)))
    positions = [(i,j) for i in range(len(GT)) for j in range(len(GT)) if i != j]
    for i,j in positions:
        # we compure the ARI between every pair of path_groundtruths
        ri = helper._rand_index_score(GT[i].flatten().tolist(),GT[j].flatten().tolist())
        matrice[i,j]=ri
    # the best groundtruth is the one maximizing sum(row) or sum(column) --- i.e. best PRI
    # the worst groundtruth is the one minimizing sum(row) or sum(column) --- i.e. worst PRI
    matrice=matrice[matrice>0].reshape((len(GT),len(GT)-1))
    best_PRI = sum(matrice[matrice.sum(axis=1).argmax()])/matrice.shape[1]
    worst_PRI = sum(matrice[matrice.sum(axis=1).argmin()])/matrice.shape[1]
    mean_PRI = numpy.mean(matrice)
    return (matrice.sum(axis=1).argmax(), matrice.sum(axis=1).argmin(), best_PRI, worst_PRI, mean_PRI)

# FIXME: COMPUTE F-MEASURE RATHER THAN PRI! SHOULD BE EASY, AND WILL MEET THE DIFFICULTY PAPER
def _stats_FMEASURE(GT):
    #print("mean of number of segments: {}".format(mean([numpy.amax(e) for e in GT])))
    stats_GT = _best_worst_groundtruth_FMEASURE(GT)
    index_best=stats_GT[0]
    index_worst=stats_GT[1]
    best_FMEASURE = stats_GT[2]
    worst_FMEASURE = stats_GT[3]
    mean_FMEASURE = stats_GT[4]
    segments = list(map(int,stats_GT[5]))
    best_groundtruth = GT[index_best]
    worst_groundtruth = GT[index_worst]
    print("number of segments {} and mean {} and stdev {}".format(segments, mean(segments), stdev(segments)))
    print("best groundtruth is {} with FMEASURE {}".format(index_best+1,best_FMEASURE))
    print("worst groundtruth is {} with FMEASURE {}".format(index_worst+1,worst_FMEASURE))
    print("mean groundtruth with FMEASURE is {}".format(mean_FMEASURE))
    return best_FMEASURE, mean_FMEASURE


# FIXME: COMPUTE F-MEASURE RATHER THAN PRI! SHOULD BE EASY, AND WILL MEET THE DIFFICULTY PAPER
def _stats_PRI(GT):
    #print("mean of number of segments: {}".format(mean([numpy.amax(e) for e in GT])))
    stats_GT = _best_worst_groundtruth_PRI(GT)
    index_best=stats_GT[0]
    index_worst=stats_GT[1]
    best_PRI = stats_GT[2]
    worst_PRI = stats_GT[3]
    mean_PRI = stats_GT[4]
    best_groundtruth = GT[index_best]
    worst_groundtruth = GT[index_worst]
    print("best groundtruth is {} with PRI {}".format(index_best+1,best_PRI))
    print("worst groundtruth is {} with PRI {}".format(index_worst+1,worst_PRI))
    print("mean groundtruth with PRI is {}".format(mean_PRI))
    return mean([numpy.amax(e) for e in GT]), best_PRI, mean_PRI

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
                gt_boundaries, gt_segmentation = helper._get_groundtruth(path_groundtruths+filename[:-4]+".mat")
                mean_segments,best_PRI, mean_PRI =_stats_PRI(gt_segmentation)
                best_FMEASURE, mean_FMEASURE =_stats_FMEASURE(gt_segmentation)
                FMEASURE.append(best_FMEASURE)
                MEAN_FMEASURE.append(mean_FMEASURE)
                PRI.append(best_PRI)
                SEGMENTS.append(mean_segments)
                MEAN_PRI.append(mean_PRI)
    print("MEAN BEST PRI, SEGMENTS, MEAN PRI\t {} {} {}".format(mean(PRI),mean(SEGMENTS),mean(MEAN_PRI)))
    print("MEAN BEST FMEASURE, MEAN FMEASURE\t {} {}".format(mean(FMEASURE),mean(MEAN_FMEASURE)))
