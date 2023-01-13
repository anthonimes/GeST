import matplotlib.pyplot as plt

from skimage import data, io
from skimage.color import rgb2gray
from skimage import segmentation, measure
from skimage.measure import label, regionprops, regionprops_table
from skimage.feature import hog
from numpy import copy, reshape

img = data.immunohistochemistry()
# SLIC result
slic = segmentation.slic(img, n_segments=200, start_label=1)
img_gs = rgb2gray(img)

# segments+1 here because otherwise regionprops always misses
# the last label
regions = measure.regionprops(slic + 1,
                              intensity_image=img_gs)
for ridx, region in enumerate(regions):
    # with mod here, because slic can sometimes create more
    # superpixel than requested. replace_samples then does
    # not have enough values, so we just start over with the
    # first one again.
    to_hog = region.intensity_image
    print(to_hog.shape)
    fd, hog_image = hog(to_hog, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, multichannel=False, feature_vector=True)
    fd=reshape(fd,(fd.shape[0]//9,9))
    print(fd.mean(axis=0))
    #io.imshow(hog_image)
    #plt.show()


