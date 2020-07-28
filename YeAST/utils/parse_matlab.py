from scipy.io import loadmat

def get_groundtruth(filepath):
    groundtruth = loadmat(filepath)
    boundaries = []
    segmentation = []
    for i in range(5):
        # groundtruths boundaries and segmentation as numpy arrays
        boundaries.append(groundtruth['groundTruth'][0][i][0]['Boundaries'][0])
        segmentation.append(groundtruth['groundTruth'][0][i][0]['Segmentation'][0])
    return boundaries, segmentation
