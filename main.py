# TEMPORARY
from os import walk, makedirs 
from src.helper import _parse_args, _savepreseg, _savefig, _colors

import warnings
warnings.filterwarnings("ignore")

import src.gest

def _write(g,arguments):
    from networkx import write_gpickle, write_weighted_edgelist
    from numpy import save
    from pickle import dump

    _spatial_radius=float(arguments['hs']) #hs
    _range_radius=float(arguments['hr']) #hr
    _min_density=int(arguments['mind']) #mind
    common=arguments['method']+"_"+str(_spatial_radius)+"_"+str(_range_radius)+"_"+str(_min_density)+"_SIGMA_"+str(arguments['sigma'])+"/"

    path_graphs = "results/graphs/"+common
    path_pickles = "results/pickles/"+common
    path_labels_msp = "results/labels/"+common
    path_labels = "results/labels/"+common
    path_presegs = "results/presegs/"+common
    path_embeddings = "results/embeddings/"+common
    path_clusterings = "results/clusterings/"+common
    path_segmentation = "results/segmentation/"+common

    makedirs(path_graphs,exist_ok=True)
    makedirs(path_pickles,exist_ok=True)
    makedirs(path_labels,exist_ok=True)
    makedirs(path_presegs,exist_ok=True)
    makedirs(path_embeddings,exist_ok=True)
    makedirs(path_clusterings,exist_ok=True)
    makedirs(path_segmentation,exist_ok=True)

    dump(g._presegmentation,open(path_labels+str(i+1)+"_"+filename[:-4]+".preseg","wb"))
    dump(g._segmentation,open(path_labels+str(i+1)+"_"+filename[:-4]+".seg","wb"))
    save(path_embeddings+filename[:-4]+".emb",g._embeddings)
    write_gpickle(g._RAG, path_pickles+str(i+1)+"_"+filename[:-4]+".pkl")
    write_weighted_edgelist(g._RAG, path_graphs+filename[:-4]+".wgt", delimiter='\t')
    _savepreseg(g._presegmentation, g._image, path_presegs+filename[:-4]+".png")
    _savefig(g._segmentation, g._image, path_segmentation+str(i+1)+"_"+filename[:-4]+"_"+str(g._number_of_regions)+".png")

def _display(g, arguments):
    import skimage.segmentation
    import skimage.color
    import skimage.measure
    import matplotlib.pyplot as plt

    if(arguments['merge']):
        fig, ax = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
    else:
        fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    ax[0][0].imshow(g._image)
    ax[0][0].set_title("initial image")
    ax[0][1].imshow(g._presegmentation)
    ax[0][1].set_title("initial segmentation")
        
    colored_regions = skimage.color.label2rgb(g._segmentation, g._image, alpha=1, colors=_colors(g._segmentation,g._image), bg_label=0)
    ax[1][0].imshow(g._segmentation)
    ax[1][0].set_title('final segmentation')
    colored_regions = skimage.color.label2rgb(g._segmentation, g._image, alpha=1, colors=_colors(g._segmentation,g._image), bg_label=0)
    ax[1][1].imshow(colored_regions)
    ax[1][1].set_title('colored final segmentation')

    if(arguments['merge']):
        ax[2][0].imshow(g._segmentation_merged)
        ax[2][0].set_title('merged segmentation')
        colored_regions = skimage.color.label2rgb(g._segmentation_merged, g._image, alpha=1, colors=_colors(g._segmentation_merged,g._image), bg_label=0)
        ax[2][1].imshow(colored_regions)
        ax[2][1].set_title('colored merged segmentation')

    # ===== START DEBUG =====
    regions = skimage.measure.regionprops(g._segmentation_merged)
    for region in regions:
        xy = region.centroid
        x = xy[1]
        y = xy[0]
        text = ax[2][0].text(x, y, region.label,ha="center", va="center", color="w")
    # ===== END DEBUG ===== 

    for a in ax.ravel():
        a.set_axis_off()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    # construct the argument parser and parse the arguments
    # common arguments
    # TODO: should maybe return the GeST instance?
    arguments = _parse_args()

    # meanshift arguments
    _spatial_radius=float(arguments['hs']) #hs
    _range_radius=float(arguments['hr']) #hr
    _min_density=int(arguments['mind']) #mind
    n_cluster = arguments['n_cluster']

    # TODO: allow for a single image as well
    dirpath,_,images = list(walk(arguments['path']))[0]

    for i,filename in enumerate(sorted(images)):
        print("===== processing image {}".format(filename))

        g = src.gest.GeST(dirpath+filename, n_cluster, preseg_method=arguments['method'],merge=arguments['merge'],contiguous=arguments['contiguous'])
        g.segmentation()
        
        # writing to hard drive or displaying with matplotlib
        if(arguments['save']): 
            _write(g,arguments)
        else:
            _display(g,arguments)
