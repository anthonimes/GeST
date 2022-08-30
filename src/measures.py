from scipy.special import comb
import numpy

def _f_measure(labels_ground_truth, labels_prediction):
    # tp = true positive, tn: true negative, fp: false positive, fn: false negative
    # number of pairs in the same set in ground truth
    sum_tp_fp = comb(numpy.bincount(labels_ground_truth), 2).sum()
    # number of pairs in the same set in prediction
    sum_tp_fn = comb(numpy.bincount(labels_prediction), 2).sum()
    # concatenating the results
    A = numpy.c_[(labels_ground_truth, labels_prediction)]
    tp = sum(comb(numpy.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in set(labels_ground_truth))
    fp = sum_tp_fp - tp
    fn = sum_tp_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return tp/(tp+(fp+fn)/2)

# np.bincount: this function is used to number of passing numbers was found.
# comb: combination example (6 2) = 15, (10, 2) = 45
# np_c: concanatenation operation.
# tp: every time a pair of elements is grouped together by the two cluster
# tn: every time a pair of elements is not grouped together by the two cluster
"""
    This function is used for calculate rand index (RI) score
    @param labels_ground_truth: actual label values
    @param labels_prediction: predicted label values
"""
def _rand_index_score(labels_ground_truth, labels_prediction):
    # tp = true positive, tn: true negative, fp: false positive, fn: false negative
    # number of pairs in the same set in ground truth
    sum_tp_fp = comb(numpy.bincount(labels_ground_truth), 2).sum()
    # number of pairs in the same set in prediction
    sum_tp_fn = comb(numpy.bincount(labels_prediction), 2).sum()
    # concatenating the results
    A = numpy.c_[(labels_ground_truth, labels_prediction)]
    tp = sum(comb(numpy.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in set(labels_ground_truth))
    fp = sum_tp_fp - tp
    fn = sum_tp_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

"""
    This function is used to applyied probabilistic rand index evaluation metric.
    @param image_name: image name
    @param prediction: slic and region merge algoritms result
    @param score / number_of_ground_truth: PRI result for related image
"""
def _probabilistic_rand_index(groundtruth, prediction):
    score = 0
    number_of_ground_truth = len(groundtruth)
    for i in range(number_of_ground_truth):
        segmentation = groundtruth[i].flatten().tolist()
        score += _rand_index_score(segmentation, prediction.flatten().tolist())
    return score / number_of_ground_truth

