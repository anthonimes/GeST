# import the necessary packages
from skimage.util import img_as_float, img_as_ubyte
from skimage.future import graph
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.spatial import distance

from skimage import io,color,measure

from utils.parse_matlab import get_groundtruth, get_BSR
from utils.graph import _color_features, _get_Lab_adjacency
from utils import save
from utils.metrics.pri import probabilistic_rand_index

from os import walk, environ, makedirs
from statistics import mean, stdev
import numpy
import networkx as nx
import warnings, sys, argparse
warnings.filterwarnings("ignore")

# https://github.com/fjean/pymeanshift
import pymeanshift as pms
# used by pymeanshift
import cv2,pickle,csv,helper

# for reproducibility
SEED = 42
environ["PYTHONHASHSEED"] = str(SEED)

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
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

    which_folder = {"val": "val/"}
    folder = which_folder[argsy['dataset']]

    path_groundtruths = path_groundtruth+folder
    path_images = argsy['path']+"/images/"+folder
    #path_impossible = argsy['path']+"/images/from_BSR"

    if method == "SLIC":
        common=method+"_"+str(_num_segments)+"_"+str(_compactness)+"_SIGMA_"+str(_sigma)+"/for_article/"
        path_labels = "results/labels/"+common
        path_figs = "results/figs/"+common
        path_figs_GT = "results/groundtruth/"+common
    else:
        common=method+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(_sigma)+"/for_article/"
        path_labels = "results/labels/"+common
        path_figs = "results/figs/"+common
        path_figs_GT = "results/groundtruth/"+common
        
    if argsy['best'] == "True":
        path_figs+="best/"
        path_figs_GT+="best/"

    makedirs(path_figs,exist_ok=True)
    makedirs(path_figs_GT,exist_ok=True)

    BEST = []
    path_article = argsy['path']+"/images/for_article"
    dirpath, dirnames, articleimages = list(walk(path_article))[0]
    
    # load the image and convert it to a floating point data type
    for (dirpath, dirnames, filenames) in walk(path_images):
        for i,filename in enumerate(sorted(filenames)):
            if filename.endswith(".jpg"):
                print("{}: {}".format(i+1,filename[:-4]))
                image = io.imread(path_image+"val/"+filename[:-4]+".jpg")
                image = img_as_float(image)
                image_lab = color.rgb2lab(image)
                image_lab = (color.rgb2lab(image) + [0,128,128]) #// [1,1,1]
                gt_boundaries, gt_segmentation = get_groundtruth(path_groundtruths+filename[:-4]+".mat")
                
                # writing all groundtruth
                for l,gt in enumerate(gt_segmentation):
                    helper._savefig(gt, image, path_figs_GT+str(i+1)+"_"+filename[:-4]+"_"+str(l+1)+".png")
                
                labels = pickle.load(open(path_labels+str(i+1)+"_"+filename[:-4]+".seg","rb"))
                helper._savefig(labels, image, path_figs+filename[:-4]+".png")
                
                labels = pickle.load(open(path_labels+str(i+1)+"_"+filename[:-4]+".preseg","rb"))
                helper._savefig(labels, image, path_figs+filename[:-4]+"_preseg.png")
