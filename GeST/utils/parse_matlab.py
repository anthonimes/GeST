from scipy.io import loadmat, savemat

def get_groundtruth(filepath):
    groundtruth = loadmat(filepath)
    boundaries = []
    segmentation = []
    for i in range(len(groundtruth['groundTruth'][0])):
        # groundtruths boundaries and segmentation as numpy arrays
        boundaries.append(groundtruth['groundTruth'][0][i][0]['Boundaries'][0])
        segmentation.append(groundtruth['groundTruth'][0][i][0]['Segmentation'][0])
    return boundaries, segmentation

def get_BSR(filepath):
    BSR = loadmat(filepath)
    return BSR['seg']

def _savemat(filepath,segmentation):
    savemat(filepath,{"segs": segmentation},appendmat=False)
