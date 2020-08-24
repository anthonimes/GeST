from skimage import data,io,color,filters,measure,util,img_as_ubyte
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from statistics import mean
import numpy

def _colors(segmentation,image):
    regions = measure.regionprops(segmentation)
    # computing masks to apply to region histograms
    # loop over the unique segment values
    colors = []
    for index,region in enumerate(regions):
        # getting coordinates of region
        coords = region.coords
        #cy, cx = region.centroid
        #plt.plot(cx, cy, 'ro')
        #plt.show()
        R_value, G_value, B_value=[],[],[]
        for (x,y) in coords:
            R,G,B=image[(x,y)]
            R_value.append(R)
            G_value.append(G)
            B_value.append(B)
        colors.append((mean(R_value),mean(G_value),mean(B_value)))
    return colors

def _savepreseg(segmentation=None,image=None,path=None,name=None):
    io.imsave(path,img_as_ubyte(mark_boundaries(image,segmentation, mode='thick')))
    #plt.axis("off")
    #plt.savefig(path,bbox_inches='tight',pad_inches=0)'''

def _savefig(segmentation=None,image=None,path=None,name=None):
    #fig = plt.figure(name)#figsize=(image.shape[1]/80,image.shape[0]/80),dpi=80)
    #if(colored):
    colored_regions = color.label2rgb(segmentation, image, alpha=1, colors=_colors(segmentation,image), bg_label=0)
    io.imsave(path,img_as_ubyte(mark_boundaries(colored_regions, segmentation, mode='thick')))
    #plt.axis("off")
    #plt.savefig(path,bbox_inches='tight',pad_inches=0)'''

def _savelabels(segmentation=None,path=None):
    with open(path, "w") as f:
        for line in segmentation:
            for label in line:
                f.write(str(label)+"\t")
            f.write("\n")

def _loadlabels(filename):
    labels = []
    with open(filename, "r") as f:
        for line in f:
            labels.append(list(map(int,line.strip().split('\t'))))
    return numpy.asarray(labels)

def _saveembeddings(embeddings,path):
     numpy.save(path,embeddings)

def _loadembeddings(path):
    return numpy.load(path)

def _savelabels_seg(segmentation=None,path=None,filename=None):
    def _compress(l):
        compressed=[]
        cpt=1
        for i in range(len(l)-1):
            if l[i] != l[i+1]:
                compressed.append((cpt,i,l[i]))
                cpt=1
            else:
                cpt+=1
        if(l[i] != l[i+1]):
            compressed.append((cpt,i,l[i]))
            compressed.append((1,i+1,l[i+1]))
        else:
            compressed.append((cpt,i+1,l[i]))
        return compressed

    with open(path, "w") as f:
        f.write('format ascii cr\n\
date XXX\n\
image '+filename+'\n\
user anthony\n\
width '+str(segmentation.shape[1])+'\n\
height '+str(segmentation.shape[0])+'\n\
segments '+str(numpy.amax(segmentation))+'\n\
gray 0\n\
invert 0\n\
flipflop 0\n\
data\n')
        for i,line in enumerate(segmentation):
            compressed=_compress(line)
            f.write(str(compressed[0][2]-1)+" "+str(i)+" 0 "+str(compressed[0][0]-1)+"\n")
            # remaining elements
            for j in range(1,len(compressed)):
                col_begin=(compressed[j][1]-compressed[j][0])+1
                col_end=col_begin+compressed[j][0]-1
                f.write(str(compressed[j][2]-1)+" "+str(i)+" "+ str(col_begin)+" "+str(col_end)+"\n")

